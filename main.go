package main

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"

	"gocv.io/x/gocv"
)

var (
	deviceID = 0
	err      error
	webcam   *gocv.VideoCapture
	// img          gocv.Mat
	haarFaceCascade gocv.CascadeClassifier
	eyeCascade      gocv.CascadeClassifier
	lbpFaceCascade  gocv.CascadeClassifier

	font = gocv.FontHersheyPlain

	// these values make sense on my Apple Studio Display's webcam, but may need
	// adjustment for other webcams
	minFaceSize = 200
	maxFaceSize = 600
)

func main() {
	if err := run(); err != nil {
		slog.Error("Exiting with error", "err", err)
		os.Exit(1)
	}
}

func run() error {
	// Open webcam
	webcam, err = gocv.OpenVideoCapture(deviceID)
	if err != nil {
		return fmt.Errorf("opening capture device %d: %w", deviceID, err)
	}
	defer webcam.Close()

	classifierPath := "/opt/homebrew/Cellar/opencv/4.9.0_4/share/opencv4"
	haarClassifierPath := filepath.Join(classifierPath, "haarcascades")
	lbpClassifierPath := filepath.Join(classifierPath, "lbpcascades")

	// Load Haar Cascade Classifier for face detection
	haarFaceCascade = gocv.NewCascadeClassifier()
	defer haarFaceCascade.Close()
	if !haarFaceCascade.Load(filepath.Join(haarClassifierPath, "haarcascade_frontalface_default.xml")) {
		return fmt.Errorf("loading Haar face classifier: %w", err)
	}

	// Load Eye Classifier
	eyeCascade = gocv.NewCascadeClassifier()
	defer eyeCascade.Close()
	if !eyeCascade.Load(filepath.Join(haarClassifierPath, "haarcascade_eye.xml")) {
		return fmt.Errorf("loading Haar eye classifier: %w", err)
	}

	// Load LBP Cascade Classifier for face detection
	lbpFaceCascade = gocv.NewCascadeClassifier()
	defer lbpFaceCascade.Close()
	if !lbpFaceCascade.Load(filepath.Join(lbpClassifierPath, "lbpcascade_frontalface_improved.xml")) {
		return fmt.Errorf("loading LBP face classifier: %w", err)
	}

	slog.Info("Server listening at http://127.0.0.1:8888/")

	// Set up HTTP server
	http.HandleFunc("/", handleRequest)
	return http.ListenAndServe("127.0.0.1:8888", nil)
}

func handleRequest(w http.ResponseWriter, r *http.Request) {
	imgMat := gocv.NewMat()
	defer imgMat.Close()

	if ok := webcam.Read(&imgMat); !ok {
		fmt.Printf("Device closed: %v\n", deviceID)
		return
	}

	// Convert to grayscale for detection
	gray := gocv.NewMat()
	defer gray.Close()
	gocv.CvtColor(imgMat, &gray, gocv.ColorBGRToGray)

	// first detect faces using the Haar frontal face classifier
	rects := haarFaceCascade.DetectMultiScale(gray)
	for _, r := range rects {
		if r.Size().X > minFaceSize && r.Size().X < maxFaceSize {
			gocv.Rectangle(&imgMat, r, color.RGBA{0, 255, 0, 0}, 2)

			sizeText := fmt.Sprintf("Size: %dx%d", r.Size().X, r.Size().Y)
			gocv.PutText(&imgMat, sizeText, image.Pt(r.Min.X, r.Min.Y-10), font, 1.0, color.RGBA{0, 255, 0, 0}, 2)

			// Detect eyes within the face region
			roiMat := imgMat.Region(r)
			defer roiMat.Close()
			eyes := eyeCascade.DetectMultiScale(roiMat)
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
	rects = lbpFaceCascade.DetectMultiScale(gray)
	for _, r := range rects {
		// if r.Size().X > minFaceSize && r.Size().X < maxFaceSize {
		gocv.Rectangle(&imgMat, r, color.RGBA{255, 0, 0, 0}, 2)

		sizeText := fmt.Sprintf("Size: %dx%d", r.Size().X, r.Size().Y)
		gocv.PutText(&imgMat, sizeText, image.Pt(r.Min.X, r.Min.Y-10), font, 1.0, color.RGBA{255, 0, 0, 0}, 2)
		// }
	}

	// Convert gocv.Mat to JPEG format
	buf, err := gocv.IMEncode(".jpg", imgMat)
	if err != nil {
		fmt.Println("Error encoding frame:", err)
		return
	}

	// Create a regular Go slice from the NativeByteBuffer
	bufSlice := make([]byte, buf.Len())
	copy(bufSlice, buf.GetBytes())

	// Create image.Image from encoded buffer
	out, _, err := image.Decode(bytes.NewReader(bufSlice))
	if err != nil {
		fmt.Println("Error decoding frame:", err)
		return
	}

	// Write image to response
	w.Header().Set("Content-Type", "image/jpeg")
	err = jpeg.Encode(w, out, nil)
	if err != nil {
		fmt.Println("Error writing image to response:", err)
	}
}
