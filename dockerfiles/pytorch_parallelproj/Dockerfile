FROM continuumio/miniconda3

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set up workspace
ENV WORKSPACE=/workspace
RUN mkdir -p $WORKSPACE
WORKDIR $WORKSPACE

# Copy requirements
COPY src/requirements.txt /tmp/requirements.txt

# Create a new conda environment
RUN conda create -y -n parallelproj_env python=3.10

COPY src/entrypoint.sh /tmp/entrypoint.sh
RUN chmod +x /tmp/entrypoint.sh

# add test data
COPY data /workspace/data

ENTRYPOINT ["/tmp/entrypoint.sh"]
CMD ["tail", "-f", "/dev/null"]