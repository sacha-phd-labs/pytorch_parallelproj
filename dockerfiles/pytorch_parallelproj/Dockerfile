FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    git \
    libsm6 \
    libxext6 \
    libxrender-dev \
    openssh-server \
    openssh-client \
    sudo \
    build-essential \
    curl \
    vim \
    nano \
    man \
    manpages \
    manpages-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Download miniconda
RUN wget https://repo.anaconda.com/archive/Anaconda3-2025.12-2-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean -afy

SHELL ["/opt/conda/bin/conda", "run", "-n", "base", "/bin/bash", "-c"]

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# update conda
RUN conda update -n base -c defaults conda -y

# Set up workspace
ENV WORKSPACE=/workspace
RUN mkdir -p $WORKSPACE
WORKDIR $WORKSPACE

# Copy requirements
COPY src/requirements.txt /tmp/requirements.txt

# Create a new conda environment
RUN conda create -y -n parallelproj_env

COPY src/entrypoint.sh /tmp/entrypoint.sh
RUN chmod +x /tmp/entrypoint.sh

# add test data
COPY data /workspace/data

# Set up SSH
RUN mkdir -p /run/sshd
RUN echo 'PermitRootLogin no' >> /etc/ssh/sshd_config
RUN echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config
RUN echo 'PubkeyAuthentication yes' >> /etc/ssh/sshd_config

# Create a default user (replace 'sacha' with your preferred username)
RUN useradd -m -s /bin/bash sacha && \
    usermod -aG sudo sacha && \
    echo "sacha:sacha" | chpasswd

EXPOSE 22

ENTRYPOINT ["/tmp/entrypoint.sh"]
CMD ["tail", "-f", "/dev/null"]
