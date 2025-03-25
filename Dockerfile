# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements1.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements1.txt
# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    python3-dev \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev && \
    rm -rf /var/lib/apt/lists/*

# Install the dlib Python package
RUN pip install dlib

# Copy the application code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]