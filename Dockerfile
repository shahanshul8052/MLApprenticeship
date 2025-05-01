# Use a lightweight Python image
FROM python:3.9-slim

# Set a working directory inside the container
WORKDIR /app

# Copy all project files to the container
COPY . /app

# Upgrade pip and install dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Default command: run the test script
CMD ["python", "test_multitask.py"]
