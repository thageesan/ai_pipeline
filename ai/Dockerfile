FROM rapidsai/rapidsai:21.06-cuda11.2-runtime-ubuntu18.04-py3.8

WORKDIR /app

COPY ./shared /app/shared
COPY ./ai /app/ai

RUN conda install pytorch gunicorn gevent pandas cudf=21.06 nltk

CMD ["gunicorn", "-w", "1", "ai.training.data:app()", "--preload", "--log-level", "debug", "--worker-class", "gevent"]