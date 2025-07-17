#!/bin/bash

docker run --gpus all --rm -it --ipc=host --name vllm-benchmarks vllm-benchmarks:latest
