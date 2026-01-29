FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Render uses this port
ENV PORT=8080
EXPOSE 8080

# Start Streamlit
CMD ["bash", "-lc", "streamlit run app_final.py --server.port $PORT --server.address 0.0.0.0"]
