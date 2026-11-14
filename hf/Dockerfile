# AfterShock on Hugging Face Spaces (Docker SDK, free CPU: 2 vCPU / 16 GB RAM).
# The code lives in the Space; the database, checkpoints and satellite tiles live in a Hugging Face
# *dataset* repo and are downloaded on start (hf/fetch_data.py).
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends fonts-dejavu-core libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face runs the container as user 1000
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY hf/requirements-hf.txt /tmp/requirements-hf.txt
RUN pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /tmp/requirements-hf.txt

COPY --chown=user . .
USER user
ENV HOME=/home/user PYTHONUNBUFFERED=1 \
    AFTERSHOCK_DATA=/home/user/bundle/data \
    AFTERSHOCK_RUNS=/home/user/bundle/runs \
    AFTERSHOCK_DB=sqlite:////home/user/bundle/data/aftershock.db

EXPOSE 7860
CMD ["bash", "hf/start.sh"]
