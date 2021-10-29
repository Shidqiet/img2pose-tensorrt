import numpy as np
from PIL import Image

import onnx
import torch.onnx
import onnxruntime
from onnxsim import simplify
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

    # input test
    x = torch.randn(1, 3, 600, 600)
    torch_out = torch_model(x)

    # export the model
    torch.onnx.export(torch_model,             # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    "onnx_model/img2pose.onnx",   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=11,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['features_0', 'features_1', 'features_2', 'features_3', 'features_pools', 
                                    'objectness', 'levels', 'proposals'])# the model's output names

    # check onnx model
    onnx_model = onnx.load("onnx_model/img2pose.onnx")
    onnx.checker.check_model(onnx_model)

    # check torch , onnx and onnx simplified model output
    # torch output
    transform = transforms.Compose([transforms.ToTensor()])
    img = Image.open('selfie.jpg')
    img = img.resize((600, 600))
    img = transform(img)
    img = img[np.newaxis, :, :, :] 
    torch_out = torch_model(img)

    # onnx output
    ort_session = onnxruntime.InferenceSession("onnx_model/img2pose.onnx")
    ort_out = ort_session.run(None, {ort_session.get_inputs()[0].name: img.numpy()})
    
    # onnx simplified output
    model = onnx.load('onnx_model/img2pose.onnx')
    model_simp, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.checker.check_model(model_simp)
    onnx.save(model_simp, 'onnx_model/img2pose_sim.onnx')
    ort_session = onnxruntime.InferenceSession("onnx_model/img2pose_sim.onnx")
    ort_sim_out = ort_session.run(None, {ort_session.get_inputs()[0].name: img.numpy()})

    # Uncomment to see
    print(torch_out)
    print(ort_out)
    print(ort_sim_out)
    
