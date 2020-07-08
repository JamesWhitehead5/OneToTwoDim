FROM tensorflow/tensorflow:latest-gpu

RUN pip install --upgrade pip && \
    pip install --no-cache-dir matplotlib