# Use official Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose Cloud Run default port
EXPOSE 8080

# Run FastAPI on port 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
