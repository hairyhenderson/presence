package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"io"
	"log/slog"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"gocv.io/x/gocv"
)

func main() {
	srv := &presenceServer{
		font: gocv.FontHersheyPlain,

		metrics: struct {
			presence prometheus.Gauge
		}{
			presence: prometheus.NewGauge(prometheus.GaugeOpts{
				Name: "presence_face_detected",
				Help: "Whether a face was detected in the last 8 seconds",
			}),
		},
	}

	defaultClassifierPath := "/usr/local/share/opencv4"
	if runtime.GOOS == "darwin" {
		defaultClassifierPath = "/opt/homebrew/Cellar/opencv/4.10.0/share/opencv4"
	}
	flag.StringVar(&srv.classifierPath, "classifier-path", defaultClassifierPath, "path to OpenCV classifier files")
	flag.StringVar(&srv.deviceID, "device", "0", "video capture device ID (numeric) or filename")
	flag.StringVar(&srv.addr, "addr", "127.0.0.1:8888", "address to listen on")
	flag.DurationVar(&srv.refreshRate, "refresh", 1*time.Second, "refresh rate for metrics and /cam handler")

	// these values make sense on my Apple Studio Display's webcam, but may need
	// adjustment for other webcams
	flag.IntVar(&srv.minFaceSize, "min-face-size", 200, "minimum face size for HAAR detection")
	flag.IntVar(&srv.maxFaceSize, "max-face-size", 600, "maximum face size for HAAR detection")

	flag.Parse()

	if err := srv.Start(srv.addr); err != nil {
		slog.Error("Exiting with error", "err", err)
		os.Exit(1)
	}
}

type presenceServer struct {
	deviceID       string
	classifierPath string
	addr           string
	refreshRate    time.Duration
	minFaceSize    int
	maxFaceSize    int

	webcam *gocv.VideoCapture

	haarFaceCascade gocv.CascadeClassifier
	eyeCascade      gocv.CascadeClassifier
	lbpFaceCascade  gocv.CascadeClassifier

	font gocv.HersheyFont

	metrics struct {
		presence prometheus.Gauge
	}

	lock sync.Mutex
}

func (srv *presenceServer) Start(addr string) error {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Open webcam
	webcam, err := gocv.OpenVideoCapture(srv.deviceID)
	if err != nil {
		return fmt.Errorf("opening capture device %q: %w", srv.deviceID, err)
	}
	defer webcam.Close()

	srv.webcam = webcam

	haarClassifierPath := filepath.Join(srv.classifierPath, "haarcascades")
	lbpClassifierPath := filepath.Join(srv.classifierPath, "lbpcascades")

	// Load Haar Cascade Classifier for face detection
	srv.haarFaceCascade = gocv.NewCascadeClassifier()
	defer srv.haarFaceCascade.Close()

	if !srv.haarFaceCascade.Load(filepath.Join(haarClassifierPath, "haarcascade_frontalface_default.xml")) {
		return fmt.Errorf("loading Haar face classifier: %w", err)
	}

	// Load Eye Classifier
	srv.eyeCascade = gocv.NewCascadeClassifier()
	defer srv.eyeCascade.Close()
	if !srv.eyeCascade.Load(filepath.Join(haarClassifierPath, "haarcascade_eye.xml")) {
		return fmt.Errorf("loading Haar eye classifier: %w", err)
	}

	// Load LBP Cascade Classifier for face detection
	srv.lbpFaceCascade = gocv.NewCascadeClassifier()
	defer srv.lbpFaceCascade.Close()
	if !srv.lbpFaceCascade.Load(filepath.Join(lbpClassifierPath, "lbpcascade_frontalface_improved.xml")) {
		return fmt.Errorf("loading LBP face classifier: %w", err)
	}

	err = prometheus.Register(srv.metrics.presence)
	if err != nil {
		return fmt.Errorf("registering metrics: %w", err)
	}

	slog.Info("Server listening", "addr", addr)

	mux := http.NewServeMux()
	mux.HandleFunc("/image.jpeg", srv.imageHandler)
	mux.HandleFunc("/cam", srv.refreshImageHandler)
	mux.HandleFunc("/presence", srv.presenceHandler)
	mux.Handle("/metrics", promhttp.Handler())

	hs := &http.Server{
		Addr:              addr,
		Handler:           mux,
		BaseContext:       func(net.Listener) context.Context { return ctx },
		ReadHeaderTimeout: 2 * time.Second,
	}

	lc := net.ListenConfig{}

	l, err := lc.Listen(ctx, "tcp", addr)
	if err != nil {
		return fmt.Errorf("listen: %w", err)
	}

	go srv.presenceCollection(ctx)

	err = hs.Serve(l)
	if err != nil && err != http.ErrServerClosed {
		return fmt.Errorf("serve: %w", err)
	}

	_ = hs.Shutdown(ctx)

	return nil
}

func (srv *presenceServer) imageHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "image/jpeg")

	if err := srv.captureImage(r.Context(), w); err != nil {
		slog.ErrorContext(r.Context(), "capturing image", "err", err)

		w.WriteHeader(http.StatusInternalServerError)
	}
}

func (srv *presenceServer) refreshImageHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "image/jpeg")
	w.Header().Set("Refresh", fmt.Sprintf("%f.0", srv.refreshRate.Seconds()))

	if err := srv.captureImage(r.Context(), w); err != nil {
		slog.ErrorContext(r.Context(), "capturing image", "err", err)

		w.WriteHeader(http.StatusInternalServerError)
	}
}

func (srv *presenceServer) captureImage(_ context.Context, w io.Writer) error {
	srv.lock.Lock()
	defer srv.lock.Unlock()

	imgMat := gocv.NewMat()
	defer imgMat.Close()

	if ok := srv.webcam.Read(&imgMat); !ok {
		return fmt.Errorf("device closed (deviceID=%q)", srv.deviceID)
	}

	// Convert to grayscale for detection
	gray := gocv.NewMat()
	defer gray.Close()
	gocv.CvtColor(imgMat, &gray, gocv.ColorBGRToGray)

	// first detect faces using the Haar frontal face classifier
	rects := srv.haarFaceCascade.DetectMultiScale(gray)
	for _, r := range rects {
		if r.Size().X > srv.minFaceSize && r.Size().X < srv.maxFaceSize {
			gocv.Rectangle(&imgMat, r, color.RGBA{0, 255, 0, 0}, 2)

			sizeText := fmt.Sprintf("Size: %dx%d", r.Size().X, r.Size().Y)
			gocv.PutText(&imgMat, sizeText, image.Pt(r.Min.X, r.Min.Y-10), srv.font, 1.0, color.RGBA{0, 255, 0, 0}, 2)

			// Detect eyes within the face region
			roiMat := imgMat.Region(r)
			defer roiMat.Close()
			eyes := srv.eyeCascade.DetectMultiScale(roiMat)
			for _, eyeRect := range eyes {
				eyeRect.Min.X += r.Min.X
				eyeRect.Min.Y += r.Min.Y
				eyeRect.Max.X += r.Min.X
				eyeRect.Max.Y += r.Min.Y
				gocv.Rectangle(&imgMat, eyeRect, color.RGBA{0, 0, 255, 0}, 2)
			}
		}
	}

	// then detect faces using the LBP frontal face classifier
	rects = srv.lbpFaceCascade.DetectMultiScale(gray)
	for _, r := range rects {
		gocv.Rectangle(&imgMat, r, color.RGBA{255, 0, 0, 0}, 2)

		sizeText := fmt.Sprintf("Size: %dx%d", r.Size().X, r.Size().Y)
		gocv.PutText(&imgMat, sizeText, image.Pt(r.Min.X, r.Min.Y-10), srv.font, 1.0, color.RGBA{255, 0, 0, 0}, 2)
	}

	// Convert gocv.Mat to JPEG format
	buf, err := gocv.IMEncode(".jpg", imgMat)
	if err != nil {
		return fmt.Errorf("frame encode: %w", err)
	}

	// Create a regular Go slice from the NativeByteBuffer
	bufSlice := make([]byte, buf.Len())
	copy(bufSlice, buf.GetBytes())

	// Create image.Image from encoded buffer
	out, _, err := image.Decode(bytes.NewReader(bufSlice))
	if err != nil {
		return fmt.Errorf("frame decode: %w", err)
	}

	err = jpeg.Encode(w, out, nil)
	if err != nil {
		return fmt.Errorf("image encode: %w", err)
	}

	return nil
}

// presenceResponse is the response to a presence request, indicating whether
// a face was detected in the image.
type presenceResponse struct {
	Present   bool `json:"present"`
	HaarFaces int  `json:"faces_haar"`
	LbpFaces  int  `json:"faces_lbp"`
	Eyes      int  `json:"eyes"`
}

func (srv *presenceServer) presenceHandler(w http.ResponseWriter, r *http.Request) {
	resp, err := srv.detect(r.Context())
	if err != nil {
		slog.ErrorContext(r.Context(), "detecting presence", "err", err)
		w.WriteHeader(http.StatusInternalServerError)

		return
	}

	w.Header().Set("Content-Type", "application/json")
	// Write presence response
	enc := json.NewEncoder(w)
	if err = enc.Encode(resp); err != nil {
		slog.ErrorContext(r.Context(), "encoding response", "err", err)

		w.WriteHeader(http.StatusInternalServerError)
	}
}

func (srv *presenceServer) presenceCollection(ctx context.Context) {
	// keep the last few presence responses with a bitmask
	// to determine the presence of a face in the last N frames
	presence := uint8(0)

	ticker := time.NewTicker(1 * time.Second)

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}

		// slog.InfoContext(ctx, "detecting presence")

		resp, err := srv.detect(ctx)
		if err != nil {
			slog.ErrorContext(ctx, "detecting presence", "err", err)
			continue
		}

		record(resp.Present, &presence)

		// if there's been any presence in the last 8 frames, set the gauge to 1
		if presence != 0 {
			srv.metrics.presence.Set(1)
		} else {
			srv.metrics.presence.Set(0)
		}
	}
}

func record(present bool, presence *uint8) {
	// shift left and set the least significant bit to 1 if present
	*presence <<= 1
	if present {
		*presence |= 1
	}
}

func (srv *presenceServer) detect(_ context.Context) (*presenceResponse, error) {
	srv.lock.Lock()
	defer srv.lock.Unlock()

	imgMat := gocv.NewMat()
	defer imgMat.Close()

	if ok := srv.webcam.Read(&imgMat); !ok {
		return nil, fmt.Errorf("device closed (deviceID=%q)", srv.deviceID)
	}

	gray := gocv.NewMat()
	defer gray.Close()

	gocv.CvtColor(imgMat, &gray, gocv.ColorBGRToGray)

	eyesFound := 0
	haarFaces := 0

	rects := srv.haarFaceCascade.DetectMultiScale(gray)
	for _, r := range rects {
		if r.Size().X > srv.minFaceSize && r.Size().X < srv.maxFaceSize {
			haarFaces++

			roiMat := imgMat.Region(r)
			defer roiMat.Close()

			eyes := srv.eyeCascade.DetectMultiScale(roiMat)
			eyesFound += len(eyes)
		}
	}

	lbpFaces := len(srv.lbpFaceCascade.DetectMultiScale(gray))

	return &presenceResponse{
		Present:   haarFaces > 0 || lbpFaces > 0,
		HaarFaces: haarFaces,
		LbpFaces:  lbpFaces,
		Eyes:      eyesFound,
	}, nil
}
