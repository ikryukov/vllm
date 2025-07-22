FROM nvcr.io/nvidia/pytorch:25.06-py3

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install basic OS deps
RUN apt-get update && apt-get install -y vim wget git curl

# Create virtual environment
RUN uv venv --python 3.12 --seed
# Activate virtual environment
RUN source .venv/bin/activate


# Create workspace directory
RUN mkdir -p /workspace/vllm

ENV MAX_JOBS=16
ARG nvcc_threads=8
ENV NVCC_THREADS=$nvcc_threads

# Make sure release wheels are built for the following architectures
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV VLLM_FA_CMAKE_GPU_ARCHES="90-real"

# Copy local vllm repository for build
COPY . /workspace/vllm

WORKDIR /workspace/vllm

RUN python3 use_existing_torch.py
RUN pip install -r requirements/build.txt
RUN pip install "numpy<2" datasets
RUN pip install --no-build-isolation -e .


ENV HF_HOME=/root/.cache/huggingface
