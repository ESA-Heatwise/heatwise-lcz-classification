# heatwise-lcz-classification: LCZ_HMSSNet training/inference EOAP processor.
#
# rasterio's pip wheel needs libexpat.so.1 at runtime, which python:3.11-slim
# doesn't ship by default (confirmed by an actual failed `docker run` here --
# see heatwise-hsi-lst-prep's Dockerfile, which hit the same issue). No
# geopandas/fiona in this repo, so libgdal-dev/gdal-bin aren't strictly
# required, but installed anyway for consistency with the other two repos'
# Dockerfiles and to avoid another round of "missing system lib" surprises.
# torch is pulled from the CPU wheel index to keep the image a reasonable
# size -- EOAP platform execution is not assumed to have GPU access; swap in
# the CUDA wheel index yourself if you need GPU training/inference in Docker.
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libexpat1 \
    libgdal-dev \
    gdal-bin \
    && rm -rf /var/lib/apt/lists/*

# cwltool (and some other CWL runners) run the container as an arbitrary
# numeric UID (--user=1000:1000 by default) with no matching /etc/passwd
# entry. torch's dynamo cache-dir resolution calls
# getpass.getuser() -> pwd.getpwuid(uid) during optimizer init (seen here:
# torch.optim.NAdam.__init__), which crashed with
# `KeyError: getpwuid(): uid not found: 1000` in an actual cwltool run.
# Pre-creating a UID-1000 user here means the lookup succeeds.
RUN useradd --create-home --uid 1000 --shell /bin/bash appuser

COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

COPY . .

# Default command for direct Docker execution.
# CWL runners override CMD with the CommandLineTool baseCommand and arguments.
CMD ["python", "/app/processor.py"]
