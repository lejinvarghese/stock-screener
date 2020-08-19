FROM python:3.5-slim
WORKDIR /usr/app

COPY ./requirements.txt .
RUN pip --default-timeout=5000 install -r requirements.txt
COPY . .

CMD ["python", "main.py"]