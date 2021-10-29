import numpy as np
from PIL import Image
import cv2

import torch
from torchvision import transforms

import sys
sys.path.append("./img2pose-modified")
from img2pose import img2poseModel
from model_loader import load_model

if __name__ == "__main__":
    # setup model 
    depth = 18
    min_size = 600
    max_size = 1400
    pose_mean = np.load('models/WIDER_train_pose_mean_v1.npy')
    pose_stddev = np.load('models/WIDER_train_pose_stddev_v1.npy')
    threed_68_points = np.load('img2pose-modified/pose_references/reference_3d_68_points_trans.npy')
    pretrained_path = 'models/img2pose_v1.pth'
    img2pose_model = img2poseModel(
        depth,
        min_size,
        max_size,
        pose_mean=pose_mean,
        pose_stddev=pose_stddev,
        threed_68_points=threed_68_points,
        device="cpu"
    )
    load_model(
        img2pose_model.fpn_model,
        pretrained_path,
        cpu_mode=True,
        model_only=True,
    )

    # get the model and make it into inference mode
    torch_model = img2pose_model.fpn_model.module
    torch_model.eval()

    # check inference
    target_size = 600
    transform = transforms.Compose([transforms.ToTensor()])
    img = Image.open('selfie.jpg')
    img = img.resize((target_size, target_size))
    img = transform(img)
    img = img[np.newaxis, :, :, :] 
    torch_out = torch_model(img)
    print(torch_out)
    