FROM python:3.8-slim
WORKDIR /pipeline
COPY requirements.txt /pipeline
RUN pip3 install -r requirements.txt
COPY FedProxiid.py /pipeline
