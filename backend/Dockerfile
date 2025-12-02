# Use Python 3.10 Slim (Lightweight & M1 Compatible)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (Required for OpenCV/Matplotlib if they need GL libraries)
# We add --no-install-recommends to keep image small
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker Cache Layer Optimization)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app (Code, Models, etc.)
COPY . .

# Expose the API port
EXPOSE 8000

# Start the server
# Note: We use 0.0.0.0 so the container listens outside itself
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]