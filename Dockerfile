# syntax=docker/dockerfile:1

# Stage 1 — build the React SPA into web/dist.
FROM node:22 AS web
WORKDIR /web
COPY web/package.json ./
# package-lock.json is gitignored upstream, so `npm install` (not `npm ci`).
RUN npm install
COPY web/ ./
RUN npm run build

# Stage 2 — Python backend serving the built SPA on a single port.
FROM python:3.12-slim
# ffmpeg: audio extraction. aria2: multi-connection downloader yt-dlp shells out to
# (absent → slower single-stream fallback, see adapters/bilibili.py).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg aria2 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.9.27 /uv /uvx /bin/

# The container runs as the host uid (compose `user:`), which has no /etc/passwd
# entry and thus no writable $HOME. Point HOME at the bind mount so torch/HF/funasr
# caches land under /data, not a read-only /.
ENV HOME=/data \
    PATH=/app/backend/.venv/bin:$PATH

# main.py resolves WEB_DIST via parents[3] → the layout must be <root>/backend + <root>/web/dist.
WORKDIR /app/backend
COPY backend/ /app/backend/
COPY --from=web /web/dist /app/web/dist

# torch variant chosen per host at install time (cpu default, cuda on NVIDIA boxes).
ARG TORCH_VARIANT=cpu
RUN uv sync --extra ${TORCH_VARIANT} --no-dev --frozen

EXPOSE 8765
CMD ["python", "-m", "bibilab.main"]
