FROM python:3.11-slim
WORKDIR /usr/app

RUN apt-get update && \
    apt-get -y install gcc curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt .
RUN pip --default-timeout=5000 install -r requirements.txt

COPY . .

ENTRYPOINT ["python", "app.py"]