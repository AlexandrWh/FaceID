FROM python:buster

WORKDIR /
EXPOSE 8000

COPY requirements.txt /requirements.txt
COPY templates /templates
COPY align_faces.py /align_faces.py
COPY anomaly_classification_v2.pt /anomaly_classification_v2.pt
COPY app.py /app.py
COPY checkpoint_10.tar /checkpoint_10.tar
COPY Dockerfile /Dockerfile
COPY models.py /models.py
COPY resnet_pipeline.py /resnet_pipeline.py
COPY utils.py /utils.py
COPY retinaface /retinaface
COPY fs_1.pt /fs_1.pt
COPY model_1.pth /model_1.pth

RUN pip install -r /requirements.txt
RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6  -y

ENTRYPOINT ["python3"]
CMD ["app.py"]