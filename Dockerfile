FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        libeccodes0 \
        libeccodes-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        tqdm \
        tensorboard \
        xarray \
        cfgrib \
        eccodes \
        scikit-optimize

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/homes/jm3320/wind-map

WORKDIR /homes/jm3320/wind-map
