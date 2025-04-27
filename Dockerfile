# Dockerfile
FROM python:3.13-slim

# avoid interactive prompts, install build-deps for GDAL/GEOS, then clean up
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      gdal-bin libgdal-dev \
      libgeos-dev \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# ensure pip is up to date, install just our Python libs
RUN pip install --no-cache-dir -U pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

# create a jovyan-like user (optional, but avoids running as root)
RUN useradd -m jovyan
USER jovyan
WORKDIR /home/jovyan

# expose notebook port
EXPOSE 8888

# start notebook, listening on all interfaces, no browser, no token (you can add a password)
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.allow_origin='*'"]
