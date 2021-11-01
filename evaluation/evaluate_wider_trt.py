import os
import time
import psutil
import logging 
from pynvml import *
from tqdm import tqdm
from collections import OrderedDict

import numpy as np
import cv2
from imgpl_utils.image import letterbox_resize

import torch
from torchvision import transforms
from torchvision.ops import boxes as box_ops
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.datasets import CocoDetection

import trt_infer_base.infer as infer_utils

import sys
sys.path.append("./img2pose-modified")
from img2pose import img2poseModel
from model_loader import load_model

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s")

# constant variable
INPUT_SHAPE = (3, 600, 600)  # channel, height, width
image_mean = [0.485, 0.456, 0.406]
image_std = [0.229, 0.224, 0.225]
depth = 18
min_size = 600
max_size = 1400
pose_mean = np.load('models/WIDER_train_pose_mean_v1.npy')
pose_stddev = np.load('models/WIDER_train_pose_stddev_v1.npy')
threed_68_points = np.load('img2pose-modified/pose_references/reference_3d_68_points_trans.npy')
pretrained_path = 'models/img2pose_v1.pth'
num_anchors_per_level = [69312, 17328, 4332, 1083, 300]

def collate_fn(batch):
    return list(zip(*batch))

def img2pose_model():
    img2pose_model = img2poseModel(
        depth,
        min_size,
        max_size,
        pose_mean=pose_mean,
        pose_stddev=pose_stddev,
        threed_68_points=threed_68_points,
    )
    load_model(
        img2pose_model.fpn_model,
        pretrained_path,
        cpu_mode=str(img2pose_model.device) == "cpu",
        model_only=True,
    )
    img2pose_model.evaluate()
    return img2pose_model

def get_roi_heads():
    # setup model 
    img2pose_model = img2poseModel(
        depth,
        min_size,
        max_size,
        pose_mean=pose_mean,
        pose_stddev=pose_stddev,
        threed_68_points=threed_68_points
    )
    load_model(
        img2pose_model.fpn_model,
        pretrained_path,
        cpu_mode=False,
        model_only=True,
    )
    # get the model and make it into inference mode
    roi_heads = img2pose_model.fpn_model.module.roi_heads
    return roi_heads

def preprocess_img(orig_img):
    orig_img = np.array(orig_img)
    img, resize_ratio, pad_w, pad_h = letterbox_resize(orig_img, 600, 600)
    # cv2.imwrite('test.jpg', infer_img)
    # exit()
    # img = cv2.resize(orig_img, (600,600))
    img = img.transpose(2, 0, 1)
    img = img.astype(np.float32)
    infer_img = np.divide(img, 255)
    # return orig_img, infer_img
    return orig_img, infer_img, 1/resize_ratio, pad_w, pad_h

def _get_top_n_idx(objectness, num_anchors_per_level):
    r = []
    offset = 0
    for ob in objectness.split(num_anchors_per_level, 1):
        num_anchors = ob.shape[1]
        pre_nms_top_n = min(6000, num_anchors)
        _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
        r.append(top_n_idx + offset)
        offset += num_anchors
    return torch.cat(r, dim=1)

def filter_proposals(proposals, objectness, image_shapes, num_anchors_per_level):
    num_images = proposals.shape[0]
    device = proposals.device
    # do not backprop throught objectness
    objectness = objectness.detach()
    objectness = objectness.reshape(num_images, -1)

    levels = [
        torch.full((n,), idx, dtype=torch.int64, device=device)
        for idx, n in enumerate(num_anchors_per_level)
    ]
    levels = torch.cat(levels, 0)
    levels = levels.reshape(1, -1).expand_as(objectness)

    # select top_n boxes independently per level before applying nms
    top_n_idx = _get_top_n_idx(objectness, num_anchors_per_level)

    image_range = torch.arange(num_images, device=device)
    batch_idx = image_range[:, None]

    objectness = objectness[batch_idx, top_n_idx]
    levels = levels[batch_idx, top_n_idx]
    proposals = proposals[batch_idx, top_n_idx]

    final_boxes = []
    for boxes, scores, lvl, img_shape in zip(
        proposals, objectness, levels, image_shapes
    ):
        boxes = box_ops.clip_boxes_to_image(boxes, img_shape)
        keep = box_ops.remove_small_boxes(boxes, 1e-3)
        boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]
        # non-maximum suppression, independently done per level
        keep = box_ops.batched_nms(boxes, scores, lvl, 0.4)
        # keep only topk scoring predictions
        keep = keep[: 1000]
        boxes, scores = boxes[keep], scores[keep]
        final_boxes.append(boxes)
    return final_boxes

def postprocess_outputs(outputs, roi_heads):
    objectness, levels, proposals = (torch.tensor(outputs['objectness'], device=torch.device('cuda')), 
                            torch.tensor(outputs['levels'], device=torch.device('cuda')), 
                            torch.tensor(outputs['proposals'], device=torch.device('cuda')))
    # objectness, proposals = (torch.tensor(outputs['objectness'], device=torch.device('cuda')), 
    #                     torch.tensor(outputs['proposals'], device=torch.device('cuda')))
    image_shapes = [INPUT_SHAPE[1:]]
    # final_boxes = filter_proposals(proposals, objectness, image_shapes, num_anchors_per_level)
    # nms for proposals from rpn
    final_boxes = []

    for boxes, scores, lvl, img_shape in zip(
        proposals, objectness, levels, image_shapes
    ):
        boxes = box_ops.clip_boxes_to_image(boxes, img_shape)
        keep = box_ops.remove_small_boxes(boxes, 1e-3)
        boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]
        # non-maximum suppression, independently done per level
        keep = box_ops.batched_nms(boxes, scores, lvl, 0.4)
        # keep only topk scoring predictions
        keep = keep[: 1000]
        boxes, scores = boxes[keep], scores[keep]
        final_boxes.append(boxes)

    # roi head
    features = OrderedDict()
    features['0'] = torch.tensor(outputs['features_0'], device=torch.device('cuda'))
    features['1'] = torch.tensor(outputs['features_1'], device=torch.device('cuda'))
    features['2'] = torch.tensor(outputs['features_2'], device=torch.device('cuda'))
    features['3'] = torch.tensor(outputs['features_3'], device=torch.device('cuda'))
    detections, detector_losses = roi_heads(
        features, final_boxes, image_shapes, None
    )

    # transform detections
    transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
    detections = transform.postprocess(
        detections, image_shapes, image_shapes
    )

    # try to check bbox
    bboxes = []
    res = detections[0]

    return detections

if __name__ == "__main__":

    process = psutil.Process(os.getpid())
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    engine_test_result = {}
    engine_test_result['img2pose engine'] = {}

    # initiate trt model
    trt_inference_wrapper = infer_utils.TRTInference(
        model_dir="onnx_model/",
        input_shape=INPUT_SHAPE,
        precision="FLOAT",
        calib_dataset=None,
        batch_size=1,
        channel_first=True,
        silent=False)
    logging.info("Initialized img2pose engine")

    # get roi heads for postprocessing
    roi_heads = get_roi_heads()

    # get data for inference
    times = []
    annFile = 'wider_face_cocoset/coco_face_validation.json'
    dataset = CocoDetection(root='wider_face_cocoset/validation', annFile=annFile)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    logging.info("Dataset loaded")
    pred_result = []

    # benchmark Time
    p_utiltime = time.time()
    trt_time = []
    cpu_usage = []
    memory_usage = []
    gpu_memory = []
    gpu_util = []

    # start benchmark
    for imgs, labels in tqdm(data_loader):
        inference_start_time = time.time()
        img = imgs[0]
        bboxes = []
        # (w, h) = img.size
        # scale_w, scale_h = w/INPUT_SHAPE[1], h/INPUT_SHAPE[2]
        # preprocess img
        orig_img, infer_img, resize_ratio, pad_w, pad_h = preprocess_img(img)
        # inference
        time1 = time.time()
        outputs = trt_inference_wrapper.infer(infer_img)
        time2 = time.time()
        times.append(time2 - time1)
        #postprocess detections
        result = postprocess_outputs(outputs, roi_heads)
        res = result[0]
        for i in range(len(res["scores"])):
            bbox = res["boxes"].cpu().numpy()[i].astype("int")
            score = res["scores"].cpu().detach().numpy()[i]
            bboxes.append(np.append(bbox, score))
        bboxes = np.asarray(bboxes)

        if np.ndim(bboxes) == 1 and len(bboxes) > 0:
                bboxes = bboxes[np.newaxis, :]

        if len(bboxes)>0:
            for box in bboxes:
                box_coco = [round((box[0]-pad_w)*resize_ratio), 
                            round((box[1]-pad_h)*resize_ratio), 
                            round(((box[2]-pad_w)*resize_ratio)-((box[0]-pad_w)*resize_ratio)), 
                            round(((box[3]-pad_h)*resize_ratio)-((box[1]-pad_h)*resize_ratio))]
                score = float(box[4])
                # if score > 0.8:
                #     cv2.rectangle(orig_img, (box_coco[0], box_coco[1]), (box_coco[0]+box_coco[2], box_coco[1]+box_coco[3]), (0, 0, 255), 2)
                pred_result.append({"image_id": int(labels[0][0]['image_id']), "category_id": 1,"bbox": box_coco, "score":score})
        # cv2.imwrite('test.jpg', orig_img)
        # exit()
        infer_time = time.time()-inference_start_time
        trt_time.append(infer_time)	
        interval = time.time()-p_utiltime
        if interval > 1:
            cpu_usage.append(process.cpu_percent())
            # get ram
            ram=int(process.memory_info().rss)
            memory_usage.append(int(ram / 1024 / 1024))
            
            mem_info = nvmlDeviceGetMemoryInfo(handle)
            res_info = nvmlDeviceGetUtilizationRates(handle)
            gpu_memory.append(int(mem_info.used / 1024 / 1024))
            gpu_util.append(res_info.gpu)
            p_utiltime=time.time()

    stat = {}

    # Benchmark Time
    stat['avg_trt_infer_time_s'] = "%.3f"%(sum(trt_time)/len(trt_time))
    print('Avg Trt Time %s s'%(stat['avg_trt_infer_time_s']))

    stat['avg_cpu_usage_per'] = "%.3f"%(sum(cpu_usage)/len(cpu_usage))
    print('Avg CPU Usage %s %%'%(stat['avg_cpu_usage_per']))

    stat['avg_mem_usage_MB'] = "%.3f"%(sum(memory_usage)/len(memory_usage))
    print('Avg Mem Usage %s MB'%(stat['avg_mem_usage_MB']))

    stat['avg_gpu_usage_per'] = "%.3f"%(sum(gpu_util)/len(gpu_util))
    print('Avg GPU Usage %s %%'%(stat['avg_gpu_usage_per']))

    stat['avg_gpumem_usage_MB'] = "%.3f"%(sum(gpu_memory)/len(gpu_memory))
    print('Avg GPU Mem Usage %s MB'%(stat['avg_gpumem_usage_MB']))

    engine_test_result['img2pose engine']['image_batch_1'] = stat

    # Export hardware data stats
    print(engine_test_result)

    # initialize COCO ground truth api, set the path of accordingly
    cocoGt = COCO(annFile)

    pred_result = [dt for dt in pred_result if dt['score'] > 0.01] #filter lower

    # load detection file
    cocoDt = cocoGt.loadRes(pred_result)

    imgIds = sorted(cocoGt.getImgIds())

    # running evaluation
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    print(f"Average time forward pass: {np.mean(np.asarray(times))}")
    trt_inference_wrapper.close()
    
