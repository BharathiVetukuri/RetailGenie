# Use a Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy script into the image
COPY train_sqlgen_t5.py /app/train_sqlgen_t5.py

# Install dependencies
RUN pip install --no-cache-dir transformers datasets pandas torch google-cloud-storage gcsfs sentencepiece

# Set the entry point
ENTRYPOINT ["python", "/app/train_sqlgen_t5.py"]

