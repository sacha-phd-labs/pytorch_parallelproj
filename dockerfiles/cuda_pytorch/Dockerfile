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

# Create a non-root devuser user (UID 1001)
ARG USERNAME=devuser
ARG USER_UID=1001
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && id -u $USERNAME || useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -s /bin/bash \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Set up workspace
ENV WORKSPACE=/workspace
RUN mkdir -p $WORKSPACE && chown $USERNAME:$USERNAME $WORKSPACE
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

# Default to non-root user
USER $USERNAME

COPY src/entrypoint.sh /tmp/entrypoint.sh
COPY src/editable_requirements.txt /tmp/editable_requirements.txt
RUN sudo chmod +x /tmp/entrypoint.sh

ENTRYPOINT ["/tmp/entrypoint.sh"]
CMD ["tail", "-f", "/dev/null"]