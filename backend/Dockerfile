# Use official Python with build tools
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for torch, pillow, numpy, etc.
RUN apt-get update && \
    apt-get install -y build-essential libjpeg-dev zlib1g-dev && \
    apt-get clean

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip setuptools && \
    pip install -r requirements.txt

# Copy the rest of your code
COPY . .

# Expose port (Railway will map it)
EXPOSE 8080
ENV PORT 8080

# Command to run your app
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080"]
