"""
Engine Utilities
QlueVision AI
Code ref: https://github.com/NVIDIA/object-detection-tensorrt-example
Copyright 2021 PT Qlue Performa Indonesia
"""

import os
import time
import warnings
import logging
from typing import Any, Dict, Callable
import numpy as np

try:
    import pycuda.driver as cuda
    import pycuda.tools

except ImportError as exc:
    warnings.warn(
        "no CUDA capable device detected", ImportWarning)

# TRT Engine creation/save/load utils
import trt_infer_base.engine as engine_utils
import tensorrt as trt

# TensorRT logger singleton
TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
STR_TO_INFERENCE_TYPE = {
    "FLOAT": trt.DataType.FLOAT,
    "HALF": trt.DataType.HALF
}

def initialize_trt_inference(
        model_dir,
        precision: str = "FLOAT",
        calib_dataset=None,
        batch_size: int = 1,
        silent: bool = True) -> tuple:
    """Check, download, and prepare package data for TRT inference.

    Parameters
    ----------
    model_dir : str
        Path to model directory containing model version
    precision : str, optional
        One of ["FLOAT", "HALF"], by default "FLOAT"
    calib_dataset : np.ndarray, optional
        dataset for int8 TensorRT calibration, by default None
    batch_size : int, optional
        Batch size of the processing, by default 1
    silent : bool, optional
        Disable printing information, by default True
        
    Returns
    -------
    tuple
        Return the loaded tensorrt.ICudaEngine and the deserialized tensorrt engine path

    Raises
    ------
    NotImplementedError
        If the requested precision type is not one of Float or Half
    """    

    trt.init_libnvinfer_plugins(TRT_LOGGER, '')

    try:
        trt_engine_datatype = STR_TO_INFERENCE_TYPE[precision]
    except KeyError:
        raise NotImplementedError(
            "Precision '%s' not yet supported!" % precision)

    # Display requested engine settings to stdout
    if not silent:
        print("TensorRT inference engine settings:")
        print("  * Inference precision - {}".format(trt_engine_datatype))
        print("  * Max batch size - {}\n".format(batch_size))

    trt_engine_path = os.path.join(model_dir, 'img2pose_sim.trt')
    onnx_model_path = os.path.join(model_dir, 'img2pose_sim.onnx')

    # If failed to get prebuilt engine, we need to build it
    if not os.path.exists(trt_engine_path) and not onnx_model_path is None:
        print("Building TensorRT engine, this might take up to ~1 min to ~45 mins...")
        # This function uses supplied .uff file
        # alongside with UffParser to build TensorRT
        # engine. For more details, check implmentation
        trt_engine = engine_utils.build_engine_from_onnx(
            onnx_model_path, TRT_LOGGER,
            trt_engine_datatype=trt_engine_datatype,
            calib_dataset=calib_dataset,
            batch_size=batch_size,
            silent=silent)
        # Save the engine to file
        engine_utils.save_engine(trt_engine, trt_engine_path)
    else:
        trt_engine = None
    if not silent:
        print('Engine is built! Ready to Use')
    return trt_engine, trt_engine_path


class TRTInference(object):
    """Manages TensorRT objects for model inference

    Parameters
    ----------
    model_dir : str
        Path to model directory containing model version
    input_shape : tuple, optional
        Shape of the input image
    precision : str
        One of ["FLOAT", "HALF"], by default "FLOAT"
    calib_dataset : np.ndarray, optional
        Dataset for int8 TensorRT calibration, by default None
    batch_size : int, optional
        Batch size of the processing, by default 1
    channel_first: bool 
        Input image as channel first [c,h,w], by default True
    silent : bool, optional
        Disable printing information, by default True

    Raises
    ------
    ValueError
        If image shape mismatch with predefined input shape
    ValueError
        If input images is 0
    ValueError
        If image batch is bigger than maximum batch
    """    

    def __init__(
            self,
            model_dir,
            input_shape=None,
            precision="FLOAT",
            calib_dataset=None,
            batch_size=1,
            channel_first=True,
            silent=True):

        # We import the auto init wrapped in class function
        cuda.init()

        # Initialize runtime needed for loading TensorRT engine from file
        self.trt_runtime = trt.Runtime(TRT_LOGGER)
        # TRT engine placeholder
        self.trt_engine = None
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.silent = silent
        self.channel_first = channel_first

        if not self.channel_first:
            if not self.silent:
                print('HWC Format is requested, please transpose input image') 
            self.input_shape = self.input_shape[1:]+(3,)

        # Initialize cuda context 
        self.cuda_ctx = pycuda.tools.make_default_context()

        self.trt_engine, trt_engine_path = initialize_trt_inference(
            model_dir=model_dir,
            precision=precision,
            calib_dataset=calib_dataset,
            batch_size=batch_size,
            silent=silent,
        )

        # If we get here, the file with engine exists, so we can load it
        if not self.trt_engine:
            deserialized_engine = engine_utils.load_engine(
                self.trt_runtime, trt_engine_path)
            self.trt_engine = engine_utils.Engine(deserialized_engine)
            if not self.silent:
                print("Loaded cached TensorRT engine from {}".format(
                    trt_engine_path))

        if isinstance(self.trt_engine, trt.tensorrt.ICudaEngine):
            self.trt_engine = engine_utils.Engine(self.trt_engine)

    def close(self):
        """Properly close the engine and detach context
        """
        self.trt_engine.close()
        if self.cuda_ctx:
            self.cuda_ctx.detach()

    @property
    def outputs(self):
        """Get list of output bindings

        Returns
        -------
        List[engine_utils.Binding]
            List of output bindings from engine_utils.Engine
        """        
        return self.trt_engine._outputs

    @property
    def max_batch_size(self):
        """Get TensorRT engine maximum batch size

        Returns
        -------
        int
            Maximum batch size of TensorRT engine can handle
        """        

        return self.trt_engine.max_batch_size

    def infer(self, image: np.ndarray) -> Any:
        """Run inference on single input image.

        Parameters
        ----------
        image: np.ndarray
            Preprocessed BGR image.
            Value range agnostic.

        Returns
        -------
        List[np.ndarray]
            List of array from TensorRT engine outputs
        """

        if image.shape != self.input_shape:
            if not self.silent:
                print('Engine input as channel first %r'%(self.channel_first))
            raise ValueError(
                "image shape mismatch with predefined input shape")

        if self.trt_engine.max_batch_size > 1:
            if not self.silent:
                print(
                    "Using 'infer' but max batch size is %d" % self.trt_engine.max_batch_size)
            return self.infer_batch(np.expand_dims(image, 0))

        # Make self the active context by pushing to top context stack
        if self.cuda_ctx:
            self.cuda_ctx.push()

        outputs = self.trt_engine.run(image)

        # Deactivate by removing self from top context stack
        if self.cuda_ctx:
            self.cuda_ctx.pop()

        # And return results
        return outputs

    def infer_multi(self, images: np.ndarray) -> Any:
        """Run inference on multiple images
        
        If the input images is bigger than the engine can handle,
        the images will be splitted and the process will run for each batch
        and the result will be stacked again according to the number of input images

        Parameters
        ----------
        images: np.ndarray
            Preprocessed (n, ) BGR image.
            Value range agnostic.

        Returns
        -------
        List[np.ndarray]
            List of array from TensorRT engine outputs
        """

        if self.trt_engine.max_batch_size is None or len(images) <= self.trt_engine.max_batch_size:
            return self.infer_batch(images)

        result = [np.zeros((len(images),) + self.outputs[x].host_buffer.shape[1:]) for x in range(len(self.outputs))]
        data_len = len(images)
        num_batches = int(data_len / self.trt_engine.max_batch_size)
        start, end = 0, 0
        for bat_i in range(num_batches):
            start, end = bat_i * \
                self.trt_engine.max_batch_size, (bat_i + 1) * \
                self.trt_engine.max_batch_size
            batch_input = images[start:end]
            output = self.infer_batch(batch_input)
            s, e = 0, 0
            for x in range(len(output)):
                s, e = bat_i * \
                    len(batch_input), (bat_i+1) * \
                    len(batch_input)
                result[x][s:e] = output[x][:len(batch_input)]
        if end < data_len:
            batch_input = images[end:]
            output = self.infer_batch(batch_input)
            result[x][end:] = output[x][:len(batch_input)]

        output_names = [output.name for output in self.outputs]

        return engine_utils.namedtupledict('outputs', output_names)(*result)

    def infer_batch(self, images: np.ndarray) -> Any:
        """Run inference on batch/multiple input images.

        Parameters
        ----------
        images: np.ndarray
            Preprocessed (n, ) BGR image.
            Value range agnostic.

        Returns
        -------
        List[np.ndarray]
            List of array from TensorRT engine outputs
        """

        # Verify if the supplied batch size is not too big
        max_batch_size = self.trt_engine.max_batch_size
        actual_batch_size = len(images)

        if not actual_batch_size:
            raise ValueError("length of images must > 0")

        arr = np.zeros((self.batch_size,) + self.input_shape, dtype=np.float32)

        # Check image shape
        if images.shape[1:] != self.input_shape:
            if not self.silent:
                print('Engine input as channel first %r'%(self.channel_first))
            raise ValueError(
                "images shape mismatch with predefined input shape")

        if actual_batch_size > max_batch_size:
            raise ValueError(
                "image batch is bigger than maximum batch")

        arr[:actual_batch_size] = images

        # Make self the active context by pushing to top context stack
        if self.cuda_ctx:
            self.cuda_ctx.push()

        outputs = self.trt_engine.run(arr)

         # Deactivate by removing self from top context stack
        if self.cuda_ctx:
            self.cuda_ctx.pop()

        return outputs
