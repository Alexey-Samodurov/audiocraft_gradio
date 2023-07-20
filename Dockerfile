FROM ubuntu:20.04

USER root

WORKDIR /app

RUN apt-get update \
    && apt-get install -y python3 python3-pip \
    && pip3 install --upgrade pip \
    && pip3 install gdown \
    && apt-get install -y ffmpeg

COPY app.py *.txt ./
COPY audiocraft ./audiocraft

ENV PYTHON_PATH=/app

RUN pip install --no-cache-dir --user -r requirements.txt

CMD ["python3", "app.py", "--server_port", "8083"]
