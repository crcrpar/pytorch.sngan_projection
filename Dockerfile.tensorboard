FROM python:3.5

RUN pip install --no-cache-dir tensorflow

WORKDIR /logs

ENTRYPOINT ["tensorboard", "--logdir", "/logs"]
CMD []
