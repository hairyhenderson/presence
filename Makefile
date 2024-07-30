.DEFAULT_GOAL = build
GO ?= go
PKG_NAME := presence
DOCKER_REPO ?= hairyhenderson/$(PKG_NAME)
PREFIX := .
# only 64-bit ARM is supported by OpenCV
DOCKER_PLATFORMS ?= linux/amd64,linux/arm64

# we just load by default, as a "dry run"
BUILDX_ACTION ?= --load
TAG_LATEST ?= latest

ifeq ("$(CI)","true")
LINT_PROCS ?= 1
else
LINT_PROCS ?= $(shell nproc)
endif

COMMIT ?= `git rev-parse --short HEAD 2>/dev/null`

COMMIT_FLAG := -X $(VERSION_PATH).GitCommit=$(COMMIT)
GO_LDFLAGS ?= $(COMMIT_FLAG) $(VERSION_FLAG)

GOOS ?= $(shell $(GO) version | sed 's/^.*\ \([a-z0-9]*\)\/\([a-z0-9]*\)/\1/')
GOARCH ?= $(shell $(GO) version | sed 's/^.*\ \([a-z0-9]*\)\/\([a-z0-9]*\)/\2/')

ifeq ("$(TARGETVARIANT)","")
ifneq ("$(GOARM)","")
TARGETVARIANT := v$(GOARM)
endif
else
ifeq ("$(GOARM)","")
GOARM ?= $(subst v,,$(TARGETVARIANT))
endif
endif

platforms := linux-amd64 linux-armv6 linux-armv7 linux-arm64 darwin-amd64 darwin-arm64

clean:
	rm -Rf $(PREFIX)/bin/*
	rm -f $(PREFIX)/*.[ci]id

%.iid: Dockerfile
	@docker build \
		--build-arg VCS_REF=$(COMMIT) \
		--target $(subst .iid,,$@) \
		--iidfile $@ \
		.

%.tag: %.iid
	@docker tag $(shell cat $<) $(DOCKER_REPO):$(TAG_LATEST)
	@echo $(DOCKER_REPO):$(TAG_LATEST) > $@

docker-multi: Dockerfile
	docker buildx build \
		--build-arg VCS_REF=$(COMMIT) \
		--platform $(DOCKER_PLATFORMS) \
		--tag $(DOCKER_REPO):$(TAG_LATEST) \
		$(BUILDX_ACTION) .

%.cid: %.iid
	@docker create $(shell cat $<) > $@

$(PREFIX)/bin/$(PKG_NAME)_%v6: $(shell find $(PREFIX) -type f -name "*.go")
	CGO_ENABLED=1 GOOS=linux GOARCH=arm GOARM=7 CC="zig cc -target arm-linux-musleabihf" CXX="zig c++ -target arm-linux-musleabihf" go build -o webcam .
	GOOS=$(shell echo $* | cut -f1 -d-) GOARCH=$(shell echo $* | cut -f2 -d- ) GOARM=6 CGO_ENABLED=$(CGO_ENABLED) \
		$(GO) build \
			-ldflags "-w -s $(GO_LDFLAGS)" \
			-o $@ \
			.

$(PREFIX)/bin/$(PKG_NAME)_%v7: $(shell find $(PREFIX) -type f -name "*.go")
	GOOS=$(shell echo $* | cut -f1 -d-) GOARCH=$(shell echo $* | cut -f2 -d- ) GOARM=7 CGO_ENABLED=$(CGO_ENABLED) \
		$(GO) build \
			-ldflags "-w -s $(GO_LDFLAGS)" \
			-o $@ \
			.

$(PREFIX)/bin/$(PKG_NAME)_%$(TARGETVARIANT): $(shell find $(PREFIX) -type f -name "*.go")
	GOOS=$(shell echo $* | cut -f1 -d-) GOARCH=$(shell echo $* | cut -f2 -d- ) GOARM=$(GOARM) CGO_ENABLED=$(CGO_ENABLED) \
		$(GO) build \
			-ldflags "-w -s $(GO_LDFLAGS)" \
			-o $@ \
			.

$(PREFIX)/bin/$(PKG_NAME): $(PREFIX)/bin/$(PKG_NAME)_$(GOOS)-$(GOARCH)$(TARGETVARIANT)
	cp $< $@

build: $(PREFIX)/bin/$(PKG_NAME)_$(GOOS)-$(GOARCH)$(TARGETVARIANT) $(PREFIX)/bin/$(PKG_NAME)

test:
	$(GO) test -race -coverprofile=c.out ./...

lint:
	@golangci-lint run --verbose --max-same-issues=0 --max-issues-per-linter=0

.PHONY: clean test build-x build lint
.DELETE_ON_ERROR:
.SECONDARY:
