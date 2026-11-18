# Multi-stage Containerfile for xrp-ai targeting RHEL 9 on Jetson Orin Nano (linux/arm64)
# Stage 1 compiles PyTorch (v2.15.0) and TorchVision (v0.29.2) from source for CUDA sm_87.
# Stage 2 uses Pure Red Hat UBI 9 + Python 3.12 Minimal for a smaller runtime image.

ARG PYTORCH_VERSION=2.15.0
ARG TORCHVISION_VERSION=0.29.2
ARG CUDA_ARCH_LIST=8.7
ARG MAX_JOBS=4
ARG INSTALL_ULTRALYTICS=0

# ==============================================================================
# Stage 1: Builder (CUDA 12.6.3 + cuDNN devel on UBI 9)
# ==============================================================================
FROM nvcr.io/nvidia/cuda:12.6.3-cudnn-devel-ubi9 AS builder

ARG PYTORCH_VERSION
ARG TORCHVISION_VERSION
ARG CUDA_ARCH_LIST
ARG MAX_JOBS
ARG INSTALL_ULTRALYTICS

USER root
WORKDIR /build

# Install Python 3.12, compiler toolchain, and EPEL 9 RPM for Stage 2 audio libs
RUN dnf install -y --setopt=install_weak_deps=False \
        python3.12 \
        python3.12-devel \
        python3.12-pip \
        git \
        gcc \
        gcc-c++ \
        make \
        which \
        zlib-devel && \
    dnf clean all && \
    curl -fsSL https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm \
        -o /tmp/epel-release-latest-9.noarch.rpm

# Create isolated Python 3.12 virtualenv and install build + project dependencies
RUN python3.12 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel && \
    /opt/venv/bin/pip install --no-cache-dir \
        cmake \
        ninja \
        numpy \
        typing_extensions \
        sympy \
        jinja2 \
        networkx \
        filelock \
        fsspec \
        pyyaml \
        packaging \
        pillow

COPY requirements.txt .
RUN /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# Build PyTorch from source for Jetson Orin (Compute Capability 8.7)
RUN git clone --depth 1 --branch "v${PYTORCH_VERSION}" --recursive --shallow-submodules \
        https://github.com/pytorch/pytorch.git /tmp/pytorch && \
    cd /tmp/pytorch && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt && \
    CUDA_HOME=/usr/local/cuda \
    CMAKE_PREFIX_PATH=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="${CUDA_ARCH_LIST}" \
    USE_CUDA=1 \
    USE_CUDNN=1 \
    USE_NCCL=0 \
    USE_DISTRIBUTED=0 \
    USE_QNNPACK=0 \
    USE_PYTORCH_QNNPACK=0 \
    BUILD_TEST=0 \
    PYTORCH_BUILD_VERSION="${PYTORCH_VERSION}" \
    PYTORCH_BUILD_NUMBER=1 \
    MAX_JOBS="${MAX_JOBS}" \
    /opt/venv/bin/python setup.py bdist_wheel && \
    /opt/venv/bin/pip install --no-cache-dir dist/torch-*.whl && \
    rm -rf /tmp/pytorch

# Build TorchVision from source against the compiled PyTorch
RUN git clone --depth 1 --branch "v${TORCHVISION_VERSION}" \
        https://github.com/pytorch/vision.git /tmp/vision && \
    cd /tmp/vision && \
    CUDA_HOME=/usr/local/cuda \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="${CUDA_ARCH_LIST}" \
    BUILD_VERSION="${TORCHVISION_VERSION}" \
    MAX_JOBS="${MAX_JOBS}" \
    /opt/venv/bin/pip install --no-cache-dir --no-build-isolation -v . && \
    rm -rf /tmp/vision

# Optionally install Ultralytics after custom torch/torchvision so pip keeps them
RUN if [ "${INSTALL_ULTRALYTICS}" = "1" ]; then \
        /opt/venv/bin/pip install --no-cache-dir ultralytics; \
    fi

# Collect only runtime CUDA 12.6 + cuDNN shared libraries (exclude static .a, stubs, headers)
RUN mkdir -p /opt/cuda-runtime-libs && \
    for dir in /usr/local/cuda/lib64 /usr/lib64; do \
        for pat in \
            'libcudart.so*' 'libcublas*.so*' 'libcufft*.so*' 'libcurand*.so*' \
            'libcusolver*.so*' 'libcusparse*.so*' 'libnvrtc*.so*' 'libnvJitLink*.so*' \
            'libcupti*.so*' 'libnvToolsExt*.so*' 'libcudnn*.so*'; do \
            cp -a "${dir}"/${pat} /opt/cuda-runtime-libs/ 2>/dev/null || true; \
        done; \
    done

# ==============================================================================
# Stage 2: Minimal Runtime (Pure UBI 9 Python 3.12 Minimal)
# ==============================================================================
FROM registry.access.redhat.com/ubi9/python-312-minimal:latest

USER root

# Install PortAudio (from EPEL 9), ALSA runtime libs (sounddevice), and OpenGL/GLib (OpenCV/YOLO)
COPY --from=builder /tmp/epel-release-latest-9.noarch.rpm /tmp/epel-release-latest-9.noarch.rpm
RUN rpm -ivh /tmp/epel-release-latest-9.noarch.rpm && \
    rm -f /tmp/epel-release-latest-9.noarch.rpm && \
    microdnf install -y --setopt=install_weak_deps=0 \
        portaudio \
        alsa-lib \
        libgomp \
        mesa-libGL \
        glib2 && \
    microdnf clean all && \
    rm -rf /var/cache/dnf /var/cache/yum

# Copy CUDA/cuDNN runtime libraries and register them with the dynamic linker
COPY --from=builder /opt/cuda-runtime-libs/ /usr/local/cuda/lib64/
RUN echo "/usr/local/cuda/lib64" > /etc/ld.so.conf.d/99-cuda.conf && ldconfig

# Copy the pre-built Python 3.12 virtualenv (project deps + custom torch & torchvision)
COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}" \
    CUDA_HOME="/usr/local/cuda" \
    HOME="/app" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    XDG_CACHE_HOME="/app/.cache"

WORKDIR /app

# Copy application code
COPY main.py requirements.txt ./
COPY nodes/ ./nodes/
COPY state/ ./state/

# Create directories for mounted runtime assets (Piper voices, WAVs, Vosk/YOLO models)
RUN mkdir -p /app/voices /app/sounds /app/models /app/.cache/vosk && \
    chown -R 1001:0 /app /opt/venv && \
    chmod -R g=u /app /opt/venv

USER 1001

ENTRYPOINT ["python", "main.py"]
CMD ["--loglevel", "INFO"]
