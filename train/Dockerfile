FROM nvcr.io/nvidia/pytorch:26.02-py3

# libsndfile for the soundfile Python package
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install only packages not already in the NGC container
# (numpy, scipy, tensorboard, pyyaml, matplotlib, tqdm are pre-installed)
RUN pip install --no-cache-dir einops pesq pystoi soundfile gguf \
    huggingface-hub librosa schedulefree

WORKDIR /workspace/deepvqe
