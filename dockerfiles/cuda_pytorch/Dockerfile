FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel AS builder

ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;12.0+PTX"
ENV FORCE_CUDA=1

WORKDIR /build

RUN apt-get update && apt-get install -y git

# Clone repo
RUN git clone https://github.com/sacha-bouchez/torch-radon.git

# Build a wheel instead of installing
WORKDIR /build/torch-radon
RUN git checkout force-architecture
RUN pip wheel . -w /build/wheels

FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    libglib2.0-0 \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    sudo \
    locales \
    && rm -rf /var/lib/apt/lists/*

# Set up workspace
ENV WORKSPACE=/workspace
RUN mkdir -p $WORKSPACE
WORKDIR $WORKSPACE

# Copy requirements (if present) early for caching
COPY src/requirements.txt /tmp/requirements.txt
RUN if [ -s /tmp/requirements.txt ]; then \
      pip --no-cache-dir install -r /tmp/requirements.txt ; \
    fi

# Copy built wheel from builder stage
COPY --from=builder /build/wheels /wheels

# Install wheel into runtime image
RUN pip install /wheels/*.whl 

# Add workspace to pythonpath
ENV PYTHONPATH="${WORKSPACE}:${PYTHONPATH}"

# Keep shell sane
ENV SHELL=/bin/bash

COPY src/entrypoint.sh /tmp/entrypoint.sh
RUN sudo chmod +x /tmp/entrypoint.sh

# add test data
COPY data /workspace/data

ENTRYPOINT ["/tmp/entrypoint.sh"]
CMD ["tail", "-f", "/dev/null"]
