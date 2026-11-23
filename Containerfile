# Multi-stage Containerfile for xrp-ai targeting RHEL 9 on Jetson Orin Nano (linux/arm64)
# Stage 1 installs Tegra (linux-aarch64) CUDA 12.6.3 + cuDNN 9.5.1 tarballs on UBI 9 and
# compiles PyTorch (v2.15.0) and TorchVision (v0.29.2) from source for CUDA sm_87.
# Stage 2 uses Pure Red Hat UBI 9 + Python 3.12 Minimal for a smaller runtime image.

ARG PYTORCH_VERSION=2.15.0
ARG TORCHVISION_VERSION=0.29.2
ARG CUDA_VERSION=12.6.3
ARG CUDNN_VERSION=9.5.1
ARG CUDA_ARCH_LIST=8.7
ARG MAX_JOBS=4
ARG INSTALL_ULTRALYTICS=1

# ==============================================================================
# Stage 1: Builder (Pure UBI 9 + Tegra linux-aarch64 CUDA 12.6.3 & cuDNN 9.5.1)
# ==============================================================================
FROM registry.access.redhat.com/ubi9/ubi:latest AS builder

ARG PYTORCH_VERSION
ARG TORCHVISION_VERSION
ARG CUDA_VERSION
ARG CUDNN_VERSION
ARG CUDA_ARCH_LIST
ARG MAX_JOBS
ARG INSTALL_ULTRALYTICS

USER root
WORKDIR /build

# Install Python 3.12, compiler toolchain, archive tools, and EPEL 9 RPM for Stage 2 audio libs
RUN dnf install -y --setopt=install_weak_deps=False \
        python3.12 \
        python3.12-devel \
        python3.12-pip \
        git \
        gcc \
        gcc-c++ \
        make \
        which \
        xz \
        tar \
        findutils \
        zlib-devel && \
    dnf clean all && \
    curl -fsSL https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm \
        -o /tmp/epel-release-latest-9.noarch.rpm

# Install Tegra (linux-aarch64) CUDA Toolkit + cuDNN tarballs into /usr/local/cuda
RUN mkdir -p /usr/local/cuda-12.6/lib && \
    ln -sfn /usr/local/cuda-12.6 /usr/local/cuda && \
    urls="$(python3.12 -c 'import json, os, urllib.request; \
get = lambda u: json.load(urllib.request.urlopen(u, timeout=60)); \
cb = "https://developer.download.nvidia.com/compute/cuda/redist/"; \
cm = get(cb + "redistrib_" + os.environ["CUDA_VERSION"] + ".json"); \
pkgs = ["cuda_cudart","cuda_nvcc","cuda_cccl","cuda_crt","cuda_nvml_dev","cuda_profiler_api","cuda_nvrtc","cuda_nvtx","cuda_cupti","cuda_cuobjdump","cuda_nvprune","cuda_culibos","libcublas","libcufft","libcurand","libcusolver","libcusparse","libnvjitlink","libcusparse_lt","libcufile"]; \
db = "https://developer.download.nvidia.com/compute/cudnn/redist/"; \
dm = get(db + "redistrib_" + os.environ["CUDNN_VERSION"] + ".json"); \
de = dm["cudnn"]["linux-aarch64"]; \
print("\n".join([cb + cm[p]["linux-aarch64"]["relative_path"] for p in pkgs if p in cm and "linux-aarch64" in cm[p]] + [db + (de["cuda12"] if "cuda12" in de else de)["relative_path"]]))')" && \
    for url in $urls; do \
        echo "Fetching $url" && \
        curl -fsSL "$url" | tar -xJf - --strip-components=1 -C /usr/local/cuda-12.6; \
    done && \
    if [ -d /usr/local/cuda-12.6/lib64 ] && [ ! -L /usr/local/cuda-12.6/lib64 ]; then \
        cp -a /usr/local/cuda-12.6/lib64/. /usr/local/cuda-12.6/lib/ && \
        rm -rf /usr/local/cuda-12.6/lib64; \
    fi && \
    ln -sfn lib /usr/local/cuda-12.6/lib64 && \
    echo "/usr/local/cuda/lib64" > /etc/ld.so.conf.d/99-cuda.conf && \
    ldconfig && \
    /usr/local/cuda/bin/nvcc --version

ENV CUDA_HOME="/usr/local/cuda" \
    PATH="/usr/local/cuda/bin:${PATH}" \
    LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs"

# Create isolated Python 3.12 virtualenv and install PyTorch/TorchVision build dependencies
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

# Build PyTorch from source for Jetson Orin (Compute Capability 8.7)
# Keep clone, build deps, and compile in separate layers so failures reuse the cached clone
RUN git clone --depth 1 --branch "v${PYTORCH_VERSION}" --recursive --shallow-submodules \
        https://github.com/pytorch/pytorch.git /tmp/pytorch

RUN /opt/venv/bin/pip install --no-cache-dir -r /tmp/pytorch/requirements.txt

RUN cd /tmp/pytorch && \
    CUDA_HOME=/usr/local/cuda \
    CMAKE_PREFIX_PATH=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="${CUDA_ARCH_LIST}" \
    USE_CUDA=1 \
    USE_CUDNN=1 \
    USE_NCCL=0 \
    USE_DISTRIBUTED=0 \
    USE_TENSORPIPE=0 \
    USE_QNNPACK=0 \
    USE_PYTORCH_QNNPACK=0 \
    USE_KINETO=0 \
    USE_FBGEMM=0 \
    USE_FLASH_ATTENTION=0 \
    USE_MEM_EFF_ATTENTION=0 \
    BUILD_TEST=0 \
    BUILD_IGNORE_SVE_UNAVAILABLE=1 \
    PYTORCH_BUILD_VERSION="${PYTORCH_VERSION}" \
    PYTORCH_BUILD_NUMBER=1 \
    MAX_JOBS="${MAX_JOBS}" \
    /opt/venv/bin/pip install --no-cache-dir --no-build-isolation -v . && \
    rm -rf /tmp/pytorch

# Build TorchVision from source against the compiled PyTorch
RUN git clone --depth 1 --branch "v${TORCHVISION_VERSION}" \
        https://github.com/pytorch/vision.git /tmp/vision

RUN cd /tmp/vision && \
    CUDA_HOME=/usr/local/cuda \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="${CUDA_ARCH_LIST}" \
    BUILD_VERSION="${TORCHVISION_VERSION}" \
    MAX_JOBS="${MAX_JOBS}" \
    /opt/venv/bin/pip install --no-cache-dir --no-build-isolation -v . && \
    rm -rf /tmp/vision

# Install project dependencies (and optionally Ultralytics) after custom torch/torchvision,
# pinning torch/torchvision so pip cannot replace them with PyPI CUDA 13 wheels
COPY requirements.txt .
RUN /opt/venv/bin/python -c "import importlib.metadata as m; print('\n'.join(f'{p}=={m.version(p)}' for p in ('torch', 'torchvision')))" > /tmp/torch-constraints.txt && \
    /opt/venv/bin/pip install --no-cache-dir -c /tmp/torch-constraints.txt -r requirements.txt && \
    if [ "${INSTALL_ULTRALYTICS}" = "1" ]; then \
        /opt/venv/bin/pip install --no-cache-dir -c /tmp/torch-constraints.txt ultralytics; \
    fi && \
    rm -f /tmp/torch-constraints.txt && \
    ! /opt/venv/bin/pip list | grep -q -- '-cu13' && \
    /opt/venv/bin/python -c "import torch, torchvision; assert torch.version.cuda and torch.version.cuda.startswith('12.6'), f'Unexpected CUDA: {torch.version.cuda}'"

# Collect only runtime Tegra CUDA 12.6 + cuDNN shared libraries (exclude static .a, stubs, headers)
RUN mkdir -p /opt/cuda-runtime-libs && \
    cp -a /usr/local/cuda/lib64/*.so* /opt/cuda-runtime-libs/ && \
    [ -z "$(find /opt/cuda-runtime-libs -xtype l)" ]

# ==============================================================================
# Stage 2: Minimal Runtime (Pure UBI 9 Python 3.12 Minimal)
# ==============================================================================
FROM registry.access.redhat.com/ubi9/python-312-minimal:latest

USER root

# Install PortAudio (from EPEL 9), ALSA runtime libs (sounddevice), libatomic (vosk), and OpenGL/GLib (OpenCV/YOLO)
COPY --from=builder /tmp/epel-release-latest-9.noarch.rpm /tmp/epel-release-latest-9.noarch.rpm
RUN rpm -ivh /tmp/epel-release-latest-9.noarch.rpm && \
    rm -f /tmp/epel-release-latest-9.noarch.rpm && \
    microdnf install -y --setopt=install_weak_deps=0 \
        portaudio \
        alsa-lib \
        libatomic \
        libgomp \
        mesa-libGL \
        glib2 \
        pipewire-utils pulseaudio-utils pipewire-alsa && \
    microdnf clean all && \
    rm -rf /var/cache/dnf /var/cache/yum

# Copy Tegra CUDA/cuDNN runtime libraries and register them with the dynamic linker
COPY --from=builder /opt/cuda-runtime-libs/ /usr/local/cuda/lib64/
COPY --from=builder /usr/local/cuda/LICENSE /usr/local/cuda/LICENSE
RUN echo "/usr/local/cuda/lib64" > /etc/ld.so.conf.d/99-cuda.conf && ldconfig

# Copy the pre-built Python 3.12 virtualenv (project deps + custom torch & torchvision)
COPY --from=builder --chown=1001:0 /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}" \
    CUDA_HOME="/usr/local/cuda" \
    HOME="/app" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    XDG_CACHE_HOME="/app/.cache"

# Verify all runtime Python packages and their shared libraries resolve in Stage 2
RUN HOME=/tmp XDG_CACHE_HOME=/tmp/.cache YOLO_CONFIG_DIR=/tmp/ultralytics \
    python -c "import vosk, sounddevice, piper, cv2, torch, torchvision, ultralytics, xrp_usb_controller" && \
    rm -rf /tmp/.cache /tmp/.config /tmp/ultralytics

WORKDIR /app

# Copy application code
COPY main.py requirements.txt ./
COPY nodes/ ./nodes/
COPY state/ ./state/

# Create directories for mounted runtime assets (Piper voices, WAVs, Vosk/YOLO models)
RUN mkdir -p /app/voices /app/sounds /app/models /app/.cache/vosk && \
    chown -R 1001:0 /app && \
    chmod -R g=u /app 

USER 1001

ENTRYPOINT ["python", "main.py"]
CMD ["--loglevel", "INFO"]
