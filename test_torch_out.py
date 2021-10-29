import numpy as np
from PIL import Image
from torchvision import transforms
import cv2
import torch

from img2pose import img2poseModel
from model_loader import load_model

if __name__ == "__main__":
    # setup model 
    depth = 18
    min_size = 600
    max_size = 600
    pose_mean = np.load('models/WIDER_train_pose_mean_v1.npy')
    pose_stddev = np.load('models/WIDER_train_pose_stddev_v1.npy')
    threed_68_points = np.load('pose_references/reference_3d_68_points_trans.npy')
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

    # # check if output same
    target_size = 600
    transform = transforms.Compose([transforms.ToTensor()])
    img = Image.open('selfie.jpg')
    img = img.resize((target_size, target_size))
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = transform(img)
    img = img[np.newaxis, :, :, :] 
    torch_out = torch_model(img)
    print(torch_out)
    
    # # try to check bbox
    # bboxes = []
    # res = torch_out[0]

    # for i in range(len(res["scores"])):
    #     bbox = res["boxes"].cpu().numpy()[i].astype("int")
    #     score = res["scores"].cpu().detach().numpy()[i]
    #     pose = res["dofs"].cpu().numpy()[i]
    #     bboxes.append(np.append(bbox, score))
    # bboxes = np.asarray(bboxes)
    # if len(bboxes)>0:
    #     order = bboxes[:, 4].ravel().argsort()[::-1]
    #     bboxes = bboxes[order, :]
    #     bboxes = bboxes[bboxes[:, 4]>0.8]
    #     for box in bboxes:
    #         cv2.rectangle(cv_img, (round(box[0]), round(box[1])), (round(box[2]), round(box[3])), (0, 0, 255), 2)
    # cv2.imwrite('test.jpg', cv_img)
    