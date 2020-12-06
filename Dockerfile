FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
MAINTAINER Dani El-Ayyass <dayyass@yandex.ru>

COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 6006
VOLUME /workspace
WORKDIR /workspace

