FROM python:3.5-slim
WORKDIR /usr/app

RUN sudo apt update && sudo apt upgrade && sudo apt install curl

COPY ./requirements.txt .
RUN pip --default-timeout=5000 install -r requirements.txt
COPY . .

CMD ["python", "app.py"]