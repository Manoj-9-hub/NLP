# Use official lightweight Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    MODEL_CHECKPOINT=t5-small \
    DATABASE_PATH=/app/models/history.db

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements standard lists
COPY requirements.txt /app/

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend app files
# We arrange files in backend/ such that the root workspace starts at nlp-cap.
# In the Docker context, we copy the files from nlp-cap/backend to /app
# We need to make sure import structures in the app match ("backend.app.xxx")
# If the root of python path is /app, we copy frontend / backend directory configurations
COPY . /app/backend/

# Change working directory so python can find backend.app module
WORKDIR /app

# Expose port
EXPOSE 8000

# Start command
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
