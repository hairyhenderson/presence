# syntax=docker/dockerfile:1.6-labs
# FROM --platform=linux/amd64 golang:1.22-alpine AS build
# FROM gocv/opencv:4.10.0-static AS presence
# FROM ghcr.io/hybridgroup/opencv:4.10.0-static AS presence
FROM ghcr.io/hybridgroup/opencv:4.10.0 AS presence

ARG TARGETOS
ARG TARGETARCH
ARG TARGETVARIANT
ENV GOOS=$TARGETOS GOARCH=$TARGETARCH

RUN apt-get update && apt-get install -y libtbb-dev
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

WORKDIR /go/src/github.com/hairyhenderson/presence
COPY go.mod /go/src/github.com/hairyhenderson/presence
COPY go.sum /go/src/github.com/hairyhenderson/presence

RUN --mount=type=cache,id=go-build-${TARGETOS}-${TARGETARCH}${TARGETVARIANT},target=/root/.cache/go-build \
	--mount=type=cache,id=go-pkg-${TARGETOS}-${TARGETARCH}${TARGETVARIANT},target=/go/pkg \
		go mod download -x

COPY . /go/src/github.com/hairyhenderson/presence

RUN --mount=type=cache,id=go-build-${TARGETOS}-${TARGETARCH}${TARGETVARIANT},target=/root/.cache/go-build \
	--mount=type=cache,id=go-pkg-${TARGETOS}-${TARGETARCH}${TARGETVARIANT},target=/go/pkg \
		make bin/presence
RUN mv bin/presence /bin/

LABEL org.opencontainers.image.revision=$VCS_REF \
	org.opencontainers.image.source="https://github.com/hairyhenderson/presence"

ENTRYPOINT [ "/bin/presence" ]
