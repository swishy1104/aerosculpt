FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DATA_DIR=/data HOST=0.0.0.0 PORT=8000
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# compile and cache the numba CFD kernels at build time so the first job starts instantly
RUN python -c "from aerosculpt.config import ensure_dirs; ensure_dirs(); from aerosculpt.jobs import warm_up; warm_up()" \
    && rm -rf /data/*
RUN useradd --create-home app && chown -R app /app && mkdir -p /data && chown app /data
USER app
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
CMD ["python", "run.py", "serve"]
