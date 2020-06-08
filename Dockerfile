FROM python:3.6.9-buster
WORKDIR /usr/src/tensorflow-ml-project
RUN apt-get update && apt-get install -y vim && \
    pip install opencv-python \
    tqdm \
    numpy \
    tensorflow \
    keras \
    matplotlib \
    pandas \
    seaborn \
    gdown \
    Pillow

COPY . .
CMD ["python", "pipeline.py"]
