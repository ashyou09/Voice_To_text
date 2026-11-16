# Runs VoxScribe on any Docker host (Render, Railway, Fly.io, Koyeb, a VPS, Hugging Face Spaces...).
# Default: light image that uses Groq for speech-to-text (set GROQ_API_KEY).
# For offline Whisper inside the container: docker build --build-arg LOCAL=1 .
FROM python:3.12-slim
ARG LOCAL=0
WORKDIR /app
COPY requirements.txt requirements-local.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    if [ "$LOCAL" = "1" ]; then pip install --no-cache-dir -r requirements-local.txt; fi
COPY . .
ENV HOST=0.0.0.0 PORT=8000
EXPOSE 8000
CMD ["python", "app.py"]
