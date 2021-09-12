import io
import flask
from flask import Flask, flash, request, redirect, url_for, render_template, jsonify, json
import flask

import math
import os
import pickle
import tarfile
import time
from glob import glob
from datetime import datetime
import json

import cv2 as cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
import faiss

from utils import align_face, get_central_face_attributes, get_all_face_attributes, draw_bboxes
from align_faces import get_reference_facial_points, warp_and_crop_face

from retinaface.detector import detect_faces


device = torch.device('cpu') #'cuda'
app = flask.Flask(__name__)

data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def load_model():
    checkpoint = 'checkpoint_10.tar'
    checkpoint = torch.load(checkpoint, map_location=device)
    
    feature_space = torch.load('fs_1.pt')
    
    global index
    global model
    global model_Yar
    global anomaly_detector 
    
    index = faiss.IndexFlatL2(feature_space.shape[1])
    index.add(feature_space)
    
    model_Yar = Model()
    model_Yar.load_state_dict(torch.load('model_1.pth'))
    model_Yar.eval()
    
    model = checkpoint['model'].module
    model = model.to(device)
    model.eval()
    
    anomaly_detector = torch.load(f'anomaly_classification_v2.pt', map_location=device)
    anomaly_detector.eval()


def detect_anomaly_Yar(image):
    border = 0.075
    
    image.thumbnail((224, 224))
    image = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])(image)
    image = torch.tensor([image.tolist()])  
    img_features = model_Yar(image)
    D, _ = index.search(img_features.detach().cpu().numpy(), 2)
    score = np.sum(D, axis=1)
        
    return False if score < border else True, score
    
    
def resnet_pipeline(model, frame, landmarks):

    landmarks = np.reshape(landmarks, (2, 5))

    transformer = data_transforms['val']

    crop_size = (112, 112)
    default_square = True
    inner_padding_factor = 0.25
    outer_padding = (0, 0)
    output_size = (112, 112)

    reference_5pts = get_reference_facial_points(
        output_size, inner_padding_factor, outer_padding, default_square)
    img = warp_and_crop_face(frame, landmarks, reference_pts=reference_5pts, crop_size=crop_size)
    img = img[..., ::-1]

    img = Image.fromarray(img, 'RGB')
    img = transformer(img)
    img = img.unsqueeze(0)
    img = img.to(device)

    output = model(img)
    descriptor = output[0].cpu().detach().numpy()

    return descriptor


def detect_face(im):
    h, w, d = im.shape
    scale = max(h/1000, w/1000)
    im = cv2.resize(im,(int(w/scale),int(h/scale)))
    landmark = detect_faces(im, confidence_threshold=0.8)
    return landmark[0], landmark[1]*scale


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def detect_anomaly(image):
    transform = transforms.Compose([
        transforms.Resize((400, 260)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    out = anomaly_detector(image).detach().cpu().numpy()[0]
    return out[1]-out[0] > 0, out[1]-out[0]


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(pretrained=False)
        self.backbone.fc = torch.nn.Identity()
        freeze_parameters(self.backbone, train_fc=True)

    def forward(self, x):
        z1 = self.backbone(x)
        z_n = F.normalize(z1, dim=-1)
        return z_n

    
def freeze_parameters(model, train_fc=False):
    for p in model.conv1.parameters():
        p.requires_grad = True
    for p in model.bn1.parameters():
        p.requires_grad = True
    for p in model.layer1.parameters():
        p.requires_grad = True
    for p in model.layer2.parameters():
        p.requires_grad = True
    if not train_fc:
        for p in model.fc.parameters():
            p.requires_grad = False
            

@app.route("/predict", methods=["POST"])
def predict():
    face_image = flask.request.files["face"].read()
    face_image = Image.open(io.BytesIO(face_image))
    face_image = np.asarray(face_image, dtype=np.uint8)
    cv2.imwrite('face_raw.jpg', face_image[:, :, ::-1])
    face_image = cv2.imread('face_raw.jpg')
    face_bboxes, face_landmarks = detect_face(face_image.copy())
    
    if len(face_landmarks) == 0:
        return json.dumps({'status': 'BAD FACE PHOTO, LANDMARKS ARE NOT DETECTED'})
    
    face_dsc = resnet_pipeline(model, face_image, face_landmarks[0])
    
    passport_image = flask.request.files["passport"].read()
    passport_image = Image.open(io.BytesIO(passport_image))
    width, height = passport_image.size
    passport_image_copy = passport_image.copy()
    passport_image = np.asarray(passport_image, dtype=np.uint8)
    cv2.imwrite('passport_raw.jpg', passport_image[:, :, ::-1])
    passport_image = cv2.imread('passport_raw.jpg')
    passport_bboxes, passport_landmarks = detect_face(passport_image.copy())
    
    if len(passport_landmarks) == 0:
        return json.dumps({'status': 'BAD PASSPORT PHOTO, LANDMARKS ARE NOT DETECTED'})
    
    passport_dsc = resnet_pipeline(model, passport_image, passport_landmarks[0])
    
    have_anomaly_Yar, anomaly_score_Yar = detect_anomaly_Yar(passport_image_copy)
    
    passport_image_copy.thumbnail((400, 260))
    (x1, y1, x2, y2) = tuple(passport_bboxes[0][:4])
    ratio = ((y2-y1)*(x2-x1))/(width*height)
    have_anomaly, anomaly_score = detect_anomaly(passport_image_copy)

    face_dsc = face_dsc/np.linalg.norm(face_dsc)
    passport_dsc = passport_dsc/np.linalg.norm(passport_dsc)
    
    return json.dumps({'score': str(np.dot(face_dsc, passport_dsc)),
                       'passport_face_landmarks':  str(passport_landmarks[0]),
                       'passport_face_detection_score': str(str(passport_bboxes[0][4])),
                       'passport_face_bbox': str(passport_bboxes[0][:4]),
                       'real_face_landmarks': str(face_landmarks[0]),
                       'real_face_bbox':  str(face_bboxes[0][:4]),
                       'real_face_detection_score': str(face_bboxes[0][4]),
                       'status': 'OK',
                       'anomaly': str(have_anomaly).lower(),
                       'anomaly_score' : str(anomaly_score),
                       'have_anomaly_Yar': str(have_anomaly_Yar).lower(),
                       'anomaly_score_Yar': str(anomaly_score_Yar[0]),
                       'potencial_incorrect_33th_page': str(ratio>0.3).lower()})
  

@app.route("/", methods=["GET"])
def home():
    data = {"success": False}

    if request.method == "GET":
        return """
        <!doctype html>
        <title>FaceID</title>
        <h1>Face verification</h1>
        <h2>Attention: Faces on photos should be vertical and frontal</h2>
        <h2>Attention: You can find face images with detected face landmarks below the cosine similarity value. Please, check, are they detected correctly.</h2>
        
        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="face" value="Upload">
            <input type="file" name="passport" value="Upload">
            <input type="submit" value="Predict">
        </form>
        """


if __name__ == "__main__":
    load_model()
    print(os.listdir('.'))
    app.run(host='0.0.0.0', port=8000)
    
    

