FROM python:3.11-slim

RUN pip install --upgrade pip

WORKDIR /agrohack2023

COPY /jupyter_nb .

COPY /data .

COPY /outputs .

COPY /nlp_model .

COPY lib .

COPY app.py .

COPY LICENSE .

COPY README.md .

COPY requirements.txt .

RUN pip install -r requirements.txt

CMD ["python", "app.py"]
