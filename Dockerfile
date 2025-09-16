FROM python:3.11-slim

# Install system deps for OpenCV video IO and ffmpeg
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only what API needs to keep image slim
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Health
ENV PORT=8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

