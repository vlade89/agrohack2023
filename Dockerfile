FROM python:3.11-slim

RUN pip install --upgrade pip

WORKDIR /agrohack2023

COPY /data .

COPY lib .

COPY clustering.py .

COPY LICENSE .

COPY main.py .

COPY README.md .

COPY requirements.txt .

RUN pip install -r requirements.txt

CMD ["python", "main.py"]
