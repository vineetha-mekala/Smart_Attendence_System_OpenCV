# ----------------------------
# Base Image (Python)
# ----------------------------
FROM python:3.10-slim

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# ----------------------------
# System Dependencies
# (Needed for OpenCV + image processing)
# ----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------
# Set Working Directory
# ----------------------------
WORKDIR /app

# ----------------------------
# Copy requirements first (best practice)
# ----------------------------
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ----------------------------
# Copy all project files
# ----------------------------
COPY . .

# ----------------------------
# Streamlit default port
# ----------------------------
EXPOSE 8501

# ----------------------------
# Run Streamlit App
# ----------------------------
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
