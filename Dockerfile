FROM python:3.9-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y ffmpeg

COPY . .

ENV PYTHON_PATH=/app

RUN pip install --no-cache-dir --user -r requirements.txt

CMD ["python3", "app.py", "--listen=0.0.0.0", "--server_port=8083"]
