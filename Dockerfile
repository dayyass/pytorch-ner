FROM python:3.7-slim-buster
MAINTAINER Dani El-Ayyass <dayyass@yandex.ru>

WORKDIR /workdir

COPY config.yaml ./
COPY data/conll2003/* data/conll2003/

RUN pip install --upgrade pip && \
    pip install --no-cache-dir pytorch-ner

CMD ["bash"]
