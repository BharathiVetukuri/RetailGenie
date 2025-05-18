FROM python:3.10-slim

WORKDIR /app

COPY train_intent_classifier.py /app/train_intent_classifier.py

# Updated to include google-cloud-storage instead of gcsfs
RUN pip install --no-cache-dir transformers pandas torch scikit-learn google-cloud-storage gcsfs sentencepiece

ENTRYPOINT ["python", "/app/train_intent_classifier.py"]
