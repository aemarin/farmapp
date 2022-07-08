FROM python:3.8

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/*
RUN wget http://archive.ubuntu.com/ubuntu/pool/main/g/glibc/multiarch-support_2.27-3ubuntu1.4_amd64.deb\
    && apt-get update \
    && apt-get install ./multiarch-support_2.27-3ubuntu1.4_amd64.deb
RUN python -m pip install -r requirements.txt
RUN pip install

EXPOSE 8080

CMD python app.py
