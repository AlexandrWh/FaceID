import math
import os
import pickle
import tarfile
import time

import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from utils import align_face, get_central_face_attributes, get_all_face_attributes, draw_bboxes
from align_faces import get_reference_facial_points, warp_and_crop_face

device = torch.device('cuda')

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



def resnet_pipeline(model, frame, landmarks):

    landmarks = np.reshape(landmarks, (2, 5))

    transformer = data_transforms['val']

    crop_size = (112, 112)
    default_square = True
    inner_padding_factor = 0.25
    outer_padding = (0, 0)
    output_size = (112, 112)

    # get the reference 5 landmarks position in the crop settings
    reference_5pts = get_reference_facial_points(
        output_size, inner_padding_factor, outer_padding, default_square)
    img = warp_and_crop_face(frame, landmarks, reference_pts=reference_5pts, crop_size=crop_size)

    img = img[..., ::-1]  # RGB
    img = Image.fromarray(img, 'RGB')  # RGB
    img = transformer(img)
    img = img.to(device)

    imgs = torch.zeros([1, 3, 112, 112], dtype=torch.float)
    imgs[0] = img

    output = model(imgs)

    descriptor = output[0].cpu().numpy()

    return descriptor

def concat_tile(im_list_2d):
    return cv.vconcat([cv.hconcat(im_list_h) for im_list_h in im_list_2d])

###EVALUATING

angles_file = 'data/angles.txt'
lfw_pickle = 'data/lfw_funneled.pkl'

with open(lfw_pickle, 'rb') as file:
    data = pickle.load(file)
samples = data['samples']
filename = 'data/lfw_test_pair.txt'
with open(filename, 'r') as file:
    lines = file.readlines()

transformer = data_transforms['val']

checkpoint = 'checkpoint_10.tar'
print(torch.cuda.is_available())
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
model = model.to(device)
model.eval()

np.random.shuffle(lines)



with torch.no_grad():
    for i, line in tqdm(enumerate(lines)):
        tokens = line.split()
        file0 = tokens[0]
        file1 = tokens[1]

        filtered0 = [sample for sample in samples if file0 in sample['full_path'].replace('\\', '/')]
        assert (len(filtered0) == 1), 'len(filtered): {} file:{}'.format(len(filtered0), file0)
        sample = filtered0[0]
        full_path0 = sample['full_path']
        landmarks0 = sample['landmarks']

        filtered1 = [sample for sample in samples if file1 in sample['full_path'].replace('\\', '/')]
        assert (len(filtered1) == 1), 'len(filtered): {} file:{}'.format(len(filtered1), file1)
        sample = filtered1[0]
        full_path1 = sample['full_path']
        landmarks1= sample['landmarks']

        frame0 = cv.imread(full_path0)
        frame1 = cv.imread(full_path1)

        descriptor0 = resnet_pipeline(model, frame0, landmarks0)
        descriptor1 = resnet_pipeline(model, frame1, landmarks1)

        x0 = descriptor0 / np.linalg.norm(descriptor0)
        x1 = descriptor1 / np.linalg.norm(descriptor1)
        cosine = np.dot(x0, x1)
        theta = math.acos(cosine)
        theta = theta * 180 / math.pi

        im_h = cv.hconcat([frame0, frame1])

        black_frame = np.zeros((50, 500, 3))

        black_frame = cv.putText(black_frame,
                                 'the same faces: {}, cosine similarity: {}'.format(bool(int(tokens[2])), cosine),
                                 (10, 40),
                                 cv.FONT_HERSHEY_SIMPLEX,
                                 0.5,
                                 (255, 255, 255),
                                 1,
                                 cv.LINE_AA)

        final_frame = np.concatenate((im_h, black_frame), axis=0)
        cv.imwrite('imgs/im{}.jpg'.format(i), final_frame)

        if i > 10:
            exit()


