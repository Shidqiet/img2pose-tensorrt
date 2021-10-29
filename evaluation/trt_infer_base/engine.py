# SPDX-License-Identifier: Apache-2.0

import warnings
import tensorrt as trt
import numpy as np
from six import string_types

try:
    import pycuda.driver
    import pycuda.gpuarray
except ImportError as exc:
    warnings.warn(
        "no CUDA capable device detected", ImportWarning)

from trt_infer_base.engine_utils import (
    check_input_validity,
    build_engine_from_onnx,
    save_engine,
    load_engine,
    namedtupledict
)

class Binding(object):
    """Create and allocate memory for engine binding

    Parameters
    ----------
    engine: tensorrt.ICudaEngine
        The deserialized TensorRT engine
    idx_or_name : str
        Index or name of the binding
    """

    def __init__(self, engine, idx_or_name):
        if isinstance(idx_or_name, string_types):
            self.name = idx_or_name
            self.index  = engine.get_binding_index(self.name)
            if self.index == -1:
                raise IndexError("Binding name not found: %s" % self.name)
        else:
            self.index = idx_or_name
            self.name  = engine.get_binding_name(self.index)
            if self.name is None:
                raise IndexError("Binding index out of range: %i" % self.index)
        self.is_input = engine.binding_is_input(self.index)


        dtype = engine.get_binding_dtype(self.index)
        dtype_map = {trt.DataType.FLOAT: np.float32,
                        trt.DataType.HALF:  np.float16,
                        trt.DataType.INT8:  np.int8,
                        trt.DataType.BOOL: np.bool}
        if hasattr(trt.DataType, 'INT32'):
            dtype_map[trt.DataType.INT32] = np.int32

        self.dtype = dtype_map[dtype]
        shape = engine.get_binding_shape(self.index)

        self.shape = tuple(shape)

        # Must allocate a buffer of size 1 for empty inputs / outputs
        if 0 in self.shape:
            self.empty = True
            # Save original shape to reshape output binding when execution is done
            self.empty_shape = self.shape
            self.shape = tuple([1])
        else:
            self.empty = False
        self._host_buf   = None
        self._device_buf = None

    @property
    def host_buffer(self):
        """Allocate a pagelocked numpy.ndarray of shape, dtype and order.

        Returns
        -------
        np.ndarray
            return pagelocked memory in host driver  
        """        
        
        if self._host_buf is None:
            self._host_buf = pycuda.driver.pagelocked_empty(self.shape, self.dtype)
        return self._host_buf

    @property
    def device_buffer(self):
        """Allocate an numpy.ndarray work-alike in device memory

        Returns
        -------
        pycuda.gpuarray.GPUArray
            the numpy.ndarray work-alike in device driver
        """        

        if self._device_buf is None:
            self._device_buf = pycuda.gpuarray.empty(self.shape, self.dtype)
        return self._device_buf

    def get_async(self, stream):
        """Transfer the contents of self into ary or a newly allocated

        Parameters
        ----------
        stream : pycuda.driver.Stream
            A streamer to handle a queue of operations that will be carried out in order

        Returns
        -------
        np.ndarray
            return pagelocked memory in host driver
        """        

        src = self.device_buffer
        dst = self.host_buffer
        src.get_async(stream, dst)
        return dst

class Engine(object):
    """Constructor for TensorRT Engine

    Parameters
    ----------
    trt_engine : tensorrt.ICudaEngine
        The deserialized TensorRT engine
    """    

    def __init__(self, trt_engine):
        self.engine = trt_engine
        nbinding = self.engine.num_bindings

        bindings = [Binding(self.engine, i)
                    for i in range(nbinding)]
        self._binding_addrs = [b.device_buffer.ptr for b in bindings]
        self._inputs  = [b for b in bindings if     b.is_input]
        self._outputs = [b for b in bindings if not b.is_input]

        for binding in self._inputs + self._outputs:
            _ = binding.device_buffer # Force buffer allocation
        for binding in self._outputs:
            _ = binding.host_buffer   # Force buffer allocation
        self.context = self.engine.create_execution_context()
        self.stream = pycuda.driver.Stream()

        output_names = [output.name for output in self._outputs]
        
    def __del__(self):
        """Delete the objects at exit
        """    

        if self.engine is not None:
            del self.engine
        if self.context is not None:
            del self.context

    def close(self):
        """Properly close the engine and detach context
        """

        self.stream.synchronize()

    @property
    def max_batch_size(self):
        """Get TensorRT engine maximum batch size

        Returns
        -------
        int
            Maximum batch size of TensorRT engine can handle
        """        

        return self.engine.max_batch_size

    def run(self, inputs, **kwargs):
        """Execute the prepared engine and return the outputs as a named tuple.
        inputs -- 

        Parameters
        ----------
        inputs : np.ndarray
            Input tensor(s) as a Numpy array or list of Numpy arrays.

        Returns
        -------
        tuple
            Tuple of output name and its corresponding np.ndarray result
        """

        if isinstance(inputs, np.ndarray):
            inputs = [inputs]

        outputs = self.execute(inputs)
        output_names = [output.name for output in self._outputs]

        return namedtupledict('outputs', output_names)(*outputs)

    def execute(self, inputs):
        """Perform inference using TensorRT

        Parameters
        ----------
        inputs : List[np.ndarray]
            Input tensor(s) as a Numpy array or list of Numpy arrays.

        Returns
        -------
        List[np.ndarray]
            List of array from TensorRT engine outputs

        Raises
        ------
        ValueError
            If the number of inputs is not enaugh for engine inputs
        """   

        if len(inputs) < len(self._inputs):
            raise ValueError("Not enough inputs. Expected %i, got %i." %
                             (len(self._inputs), len(inputs)))
        if isinstance(inputs, dict):
            inputs = [inputs[b.name] for b in self._inputs]

        for i, (input_array, input_binding) in enumerate(zip(inputs, self._inputs)):
            input_array = np.expand_dims(input_array, axis=0)
            input_array = np.array(input_array, dtype=input_array.dtype, order='C')
            input_array = check_input_validity(i, input_array, input_binding)
            input_binding_array = input_binding.device_buffer
            input_binding_array.set_async(input_array, self.stream)

        self.context.execute_async_v2(
            self._binding_addrs, self.stream.handle)

        results = [output.get_async(self.stream)
                   for output in self._outputs]

        # For any empty bindings, update the result shape to the expected empty shape
        for i, (output_array, output_binding) in enumerate(zip(results, self._outputs)):
            if output_binding.empty:
                results[i] = np.empty(shape=output_binding.empty_shape, dtype=output_binding.dtype)

        self.stream.synchronize()

        return results
