FROM python:3.9-slim
WORKDIR /usr/app

RUN apt-get update 
RUN apt-get -y install gcc
RUN apt-get -y install curl

COPY ./requirements.txt .
RUN pip --default-timeout=5000 install -r requirements.txt
COPY . .

ENTRYPOINT ["python", "app.py"]