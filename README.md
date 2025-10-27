# 🐳🦭 Container Setup: Docker & Podman

Below are **copy‑paste ready** commands for both **Docker** and **Podman** to build, push, and run the Compression API image.

---

## Docker

### Build (for RunPod / AMD64)
```bash
docker buildx build --platform linux/amd64   -t <DOCKERHUB_USER>/compression-api:v0.1.0 . --load
```

### Run Locally
Prepare directories:
```bash
mkdir -p uploads outputs
```
Run container:
```bash
docker run --rm -p 8000:8000   -v "$PWD/uploads":/app/uploads   -v "$PWD/outputs":/app/outputs   <DOCKERHUB_USER>/compression-api:v0.1.0
```
Open: http://localhost:8000/docs

---

## Podman

### Build (for RunPod / AMD64)
> Works on Linux/macOS with Podman 4+. Tags to Docker Hub namespace.
```bash
podman build --platform linux/amd64   -t docker.io/<DOCKERHUB_USER>/compression-api:v0.1.0 .
```

### (Optional) Multi‑arch Build & Push (Manifest)
> Podman uses a **manifest** to publish a single tag for multiple architectures.

Build per-arch images locally:
```bash
podman build --platform linux/amd64    -t localhost/compression-api:amd64 .
```
