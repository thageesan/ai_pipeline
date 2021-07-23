FROM python:3.9-slim

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    libpoppler-cpp-dev \
    pkg-config \
    python3-dev \
    gcc \
    git \
    libncurses5

WORKDIR /app

COPY ./shared /app/shared
COPY ./ai /app/ai

RUN pip install pyemd pandas nltk py_stringmatching==0.4.2 gputil num2words gensim transformers torch google-api-python-client oauth2client
RUN git clone https://github.com/epfml/sent2vec.git && cd sent2vec && make && pip install --upgrade cython && pip install . && cd .. && rm -rf sent2vec

ENTRYPOINT ["/bin/bash"]