# Use a small, modern Python image
FROM python:3.11-slim

# Create app directory
WORKDIR /app

# Prevent python from writing .pyc files and ensure output appears in logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system deps required by some packages (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements-server.txt /app/requirements-server.txt
RUN pip install --no-cache-dir -r /app/requirements-server.txt

# Copy application code
COPY src /app/src
# Copy model artifacts - you must have 'models' folder in build context
COPY models /app/models

# Expose port (we will run uvicorn on port 80)
EXPOSE 80

ENV MODEL_PATH=/app/models/rf_model.joblib
ENV LABEL_ENCODER_PATH=/app/models/label_encoder.joblib
ENV META_PATH=/app/models/meta.json

# Run uvicorn (use a production workers count if desired)
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "80", "--workers", "1"]
