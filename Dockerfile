# ---------- Stage 1: Build FFmpeg with libvmaf ----------
FROM ubuntu:24.04 AS ffmpeg-build
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential pkg-config yasm nasm meson ninja-build cmake \
    curl ca-certificates wget xxd python3 python3-pip \
    libx264-dev libx265-dev libnuma-dev libvpx-dev libaom-dev \
    libfreetype6-dev libfribidi-dev libass-dev libmp3lame-dev libopus-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/src

# Build libvmaf
RUN git clone --depth=1 https://github.com/Netflix/vmaf.git && \
    cd vmaf/libvmaf && meson setup build --buildtype release -Denable_float=true && \
    ninja -C build && ninja -C build install

# Build FFmpeg with libvmaf
ARG FFMPEG_TAG=n7.0.2
RUN git clone --depth=1 --branch ${FFMPEG_TAG} https://github.com/FFmpeg/FFmpeg.git ffmpeg && \
    cd ffmpeg && ./configure \
        --prefix=/opt/ffmpeg \
        --pkg-config-flags="--static" \
        --extra-cflags="-I/usr/local/include" \
        --extra-ldflags="-L/usr/local/lib" \
        --extra-libs="-lpthread -lm" \
        --bindir=/opt/ffmpeg/bin \
        --enable-gpl \
        --enable-libx264 --enable-libx265 --enable-libvpx \
        --enable-libopus --enable-libmp3lame --enable-libass --enable-libfreetype \
        --enable-libvmaf \
    && make -j"$(nproc)" && make install


# ---------- Stage 2: Runtime (API only) ----------
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LIBVMAF_MODEL_PATH=/usr/share/vmaf/model \
    PATH=/opt/ffmpeg/bin:$PATH

# Core runtime + Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip ca-certificates git curl tini \
 && rm -rf /var/lib/apt/lists/*

# Codec runtime libs FFmpeg expects at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libass9 libfreetype6 libfribidi0 libmp3lame0 libopus0 libnuma1 \
    libx264-164 libx265-199 libvpx9 libaom3 \
 && rm -rf /var/lib/apt/lists/*

# Bring FFmpeg + libvmaf shared libs from builder
COPY --from=ffmpeg-build /opt/ffmpeg /opt/ffmpeg
COPY --from=ffmpeg-build /usr/local/lib/ /usr/local/lib/

# Make libvmaf discoverable
RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/ffmpeg.conf && ldconfig

# VMAF models
RUN mkdir -p /usr/share/vmaf && \
    git clone --depth=1 https://github.com/Netflix/vmaf.git /tmp/vmaf && \
    cp -r /tmp/vmaf/model /usr/share/vmaf/ && rm -rf /tmp/vmaf

# App
RUN useradd -ms /bin/bash appuser
WORKDIR /app

# Copy your project (keep Dockerfile in the same folder as app.py)
COPY . /app

# Create writable directories for THIS repo
RUN mkdir -p /app/uploads /app/outputs /app/results /app/logs && \
    chown -R appuser:appuser /app


# Python deps
RUN python3 -m venv /opt/venv && . /opt/venv/bin/activate && \
    pip install --upgrade pip wheel setuptools && \
    if [ -s requirements.txt ]; then pip install -r requirements.txt; fi
ENV PATH="/opt/venv/bin:${PATH}"

USER appuser
EXPOSE 8000
ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000"]
