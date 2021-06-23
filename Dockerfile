FROM kaldiasr/kaldi:2020-09

ENV KALDI_ROOT=/opt/kaldi
ENV PATH=/opt/kaldi/src/bin:$PATH

RUN apt-get update && apt-get install -y python3-pip

WORKDIR /wrk

COPY requirements.txt ./

RUN pip3 install -r requirements.txt

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.5 10

RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 10

COPY recipe/ ./recipe/

WORKDIR /wrk/recipe
