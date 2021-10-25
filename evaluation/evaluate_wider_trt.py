import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.datasets import CocoDetection
from tqdm import tqdm

import sys
sys.path.append("./")
from img2pose import img2poseModel
from model_loader import load_model

import torch.onnx
import onnx
import onnxruntime
from onnxsim import simplify

# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval

import trt_infer_base.infer as infer_utils

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

    # # input test
    # x = torch.randn(1, 3, 600, 600)
    # torch_out = torch_model(x)

    # # # # export the model
    # # # torch.onnx.export(torch_model,             # model being run
    # # #                 x,                         # model input (or a tuple for multiple inputs)
    # # #                 "evaluation/model/img2pose/img2pose.onnx",   # where to save the model (can be a file or file-like object)
    # # #                 export_params=True,        # store the trained parameter weights inside the model file
    # # #                 opset_version=11,          # the ONNX version to export the model to
    # # #                 do_constant_folding=True,  # whether to execute constant folding for optimization
    # # #                 input_names = ['input'],   # the model's input names
    # # #                 output_names = ['bboxes', 'labels', 'scores', 'dofs'],# the model's output names
    # # #                 dynamic_axes={'bboxes' : [0], 'labels': [0], 'scores': [0], 'dofs': [0]})
    # # export the model
    # torch.onnx.export(torch_model,             # model being run
    #                 x,                         # model input (or a tuple for multiple inputs)
    #                 "evaluation/model/img2pose/img2pose.onnx",   # where to save the model (can be a file or file-like object)
    #                 export_params=True,        # store the trained parameter weights inside the model file
    #                 opset_version=11,          # the ONNX version to export the model to
    #                 do_constant_folding=True,  # whether to execute constant folding for optimization
    #                 input_names = ['input'],   # the model's input names
    #                 output_names = ['proposals', 'proposals_scores'])# the model's output names

    # # # check onnx model
    # onnx_model = onnx.load("evaluation/model/img2pose/img2pose.onnx")
    # onnx.checker.check_model(onnx_model)

    # # check if output same
    transform = transforms.Compose([transforms.ToTensor()])
    img = Image.open('selfie.jpg')
    img = img.resize((600, 600))
    img = transform(img)
    img = img[np.newaxis, :, :, :] 
    torch_out = torch_model(img)
    print(torch_out[1].size())
    exit()

    # ort_session = onnxruntime.InferenceSession("evaluation/model/img2pose/img2pose.onnx")
    # ort_out = ort_session.run(None, {ort_session.get_inputs()[0].name: img.numpy()})
    
    # # simplify model
    # model = onnx.load('evaluation/model/img2pose/img2pose.onnx')
    # model_simp, check = simplify(model)
    # assert check, "Simplified ONNX model could not be validated"
    # # check onnx model
    # onnx.save(model_simp, 'evaluation/model/img2pose/img2pose_sim.onnx')
    # onnx_model = onnx.load("evaluation/model/img2pose/img2pose_sim.onnx")
    # onnx.checker.check_model(onnx_model)

    # ort_session = onnxruntime.InferenceSession("evaluation/model/img2pose/img2pose_sim.onnx")
    # ort_out = ort_session.run(None, {ort_session.get_inputs()[0].name: img.numpy()})

    import cv2

    image = cv2.imread('selfie.jpg')
    image = cv2.resize(image, (600,600))
    image = image.transpose(2, 0, 1)
    image = image.astype(np.float32)
    INPUT_SHAPE = (3, 600, 600)  # channel, height, width
    trt_inference_wrapper = infer_utils.TRTInference(
        model_dir="evaluation/model/",
        input_shape=INPUT_SHAPE,
        precision="FLOAT",
        calib_dataset=None,
        batch_size=1,
        channel_first=True,
        silent=False)

    outputs = trt_inference_wrapper.infer(image)
    print(outputs)
    trt_inference_wrapper.close()
    
