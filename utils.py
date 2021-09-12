import argparse
import logging
import math

import cv2 as cv
import numpy as np
import torch
from PIL import Image

from align_faces import get_reference_facial_points, warp_and_crop_face


image_h, image_w = 112, 112

def align_face(img_fn, facial5points):
    raw = cv.imread(img_fn, True)   # BGR
    facial5points = np.reshape(facial5points, (2, 5))

    crop_size = (image_h, image_w)

    default_square = (image_h == image_w)
    inner_padding_factor = 0.25
    outer_padding = (0, 0)
    output_size = (image_h, image_w)

    # get the reference 5 landmarks position in the crop settings
    reference_5pts = get_reference_facial_points(
        output_size, inner_padding_factor, outer_padding, default_square)

    # dst_img = warp_and_crop_face(raw, facial5points)
    dst_img = warp_and_crop_face(raw, facial5points, reference_pts=reference_5pts, crop_size=crop_size)
    return dst_img


def get_face_attributes(full_path):
    try:
        img = Image.open(full_path).convert('RGB')
        bounding_boxes, landmarks = detect_faces(img)

        if len(landmarks) > 0:
            landmarks = [int(round(x)) for x in landmarks[0]]
            return True, landmarks

    except KeyboardInterrupt:
        raise
    except:
        pass
    return False, None


def select_central_face(im_size, bounding_boxes):
    width, height = im_size
    nearest_index = -1
    nearest_distance = 100000
    for i, b in enumerate(bounding_boxes):
        x_box_center = (b[0] + b[2]) / 2
        y_box_center = (b[1] + b[3]) / 2
        x_img = width / 2
        y_img = height / 2
        distance = math.sqrt((x_box_center - x_img) ** 2 + (y_box_center - y_img) ** 2)
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_index = i

    return nearest_index


def get_central_face_attributes(full_path):
    try:
        img = Image.open(full_path).convert('RGB')
        bounding_boxes, landmarks = detect_faces(img)

        if len(landmarks) > 0:
            i = select_central_face(img.size, bounding_boxes)
            return True, [bounding_boxes[i]], [landmarks[i]]

    except KeyboardInterrupt:
        raise
    except:
        pass
    return False, None, None


def get_all_face_attributes(full_path):
    img = Image.open(full_path).convert('RGB')
    bounding_boxes, landmarks = detect_faces(img)
    return bounding_boxes, landmarks


def draw_bboxes(img, bounding_boxes, facial_landmarks=[]):
    for b in bounding_boxes:
        cv.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255), 1)

    for p in facial_landmarks:
        for i in range(5):
            cv.circle(img, (int(p[i]), int(p[i + 5])), 1, (0, 255, 0), -1)

        break  # only first

    return img



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

import torch
from PIL import Image
import torchvision.transforms as transforms
import faiss
import torchvision.models as models
import torch.nn.functional as F
import utils

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet152(pretrained=False)
        self.backbone.fc = torch.nn.Identity()
        freeze_parameters(self.backbone, train_fc=True)

    def forward(self, x):
        z1 = self.backbone(x)
        z_n = F.normalize(z1, dim=-1)
        return z_n