FROM python:3.10.11-slim-buster

WORKDIR /app

COPY . .

RUN pip install --upgrade pip \
    && pip install --default-timeout=100 --no-cache-dir -r requirements.txt

ENV PORT=5000
EXPOSE 5000

CMD ["python", "app.py"]