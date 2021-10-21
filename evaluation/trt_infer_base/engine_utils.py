import numpy as np
import onnx
import onnx_graphsurgeon as gs
from six import string_types
from collections import namedtuple
import tensorrt as trt

def check_input_validity(input_idx, input_array, input_binding):
    """Check and validate input according to engine's inputs

    Parameters
    ----------
    input_idx : int
        Binding idx
    input_array : np.ndarray
        The input that will be checked
    input_binding : engine.Binding
        Binding object
        
    Returns
    -------
    np.ndarray
        The array that passed engine check

    Raises
    ------
    ValueError
        If the input shape is different than engine's expected
    TypeError
        If the given dtype is different than engine's expected. Cannot safely cast.
    """

    # Check shape
    trt_shape = tuple(input_binding.shape)
    onnx_shape    = tuple(input_array.shape)
    
    if onnx_shape != trt_shape:
        if not (trt_shape == (1,) and onnx_shape == ()) :
            raise ValueError("Wrong shape for input %i. Expected %s, got %s." %
                            (input_idx, trt_shape, onnx_shape))

    # Check dtype
    if input_array.dtype != input_binding.dtype:
        #TRT does not support INT64, need to convert to INT32
        if input_array.dtype == np.int64 and input_binding.dtype == np.int32:
            casted_input_array = np.array(input_array, copy=True, dtype=np.int32)
            if np.equal(input_array, casted_input_array).all():
                input_array = casted_input_array
            else:
                raise TypeError("Wrong dtype for input %i. Expected %s, got %s. Cannot safely cast." %
                            (input_idx, input_binding.dtype, input_array.dtype))
        else:
            raise TypeError("Wrong dtype for input %i. Expected %s, got %s." %
                            (input_idx, input_binding.dtype, input_array.dtype))
    return input_array

def modify_onnx_model(onnx_path, batch_size):
    """Modify onnx model batch size

    Parameters
    ----------
    onnx_path : str
        path to load onnx model
    batch_size : int
        desired batch size

    Returns
    -------
    onnx.ModelProto
        modified onnx model
    """
    graph = gs.import_onnx(onnx.load(onnx_path))

    # Change the input and output batch size
    for inp in graph.inputs:
        inp.shape[0] = batch_size
    for out in graph.outputs:
        out.shape[0] = batch_size

    # Make sure input dtype
    for inp in graph.inputs:
        inp.dtype = np.float32

    graph = graph.cleanup().toposort()

    return gs.export_onnx(graph)

def build_engine_from_onnx(
        onnx_model_path,
        trt_logger,
        trt_engine_datatype=trt.DataType.FLOAT,
        calib_dataset=None,
        batch_size=1,
        silent=True):
    """Build TensorRT engine from ONNX model

    Parameters
    ----------
    onnx_model_path : str
        Path to load ONNX model
    trt_logger :  tensorrt.Logger
        TensorRT Logger for the Builder, ICudaEngine and Runtime
    trt_engine_datatype : tensorrt.DataType, optional
        The requested engine type, by default trt.DataType.FLOAT
    calib_dataset : np.ndarray, optional
        Dataset for int8 TensorRT calibration, by default None
    batch_size : int, optional
        Batch size of the processing, by default 1
    silent : bool, optional
        Disable printing information, by default True

    Returns
    -------
    tensorrt.ICudaEngine
        A new ICudaEngine

    Raises
    ------
    NotImplementedError
        If the requested precision type is not one of Float or Half
    RuntimeError
        If error found while parsing node from ONNX to TensorRT
    """  
          
    explicit_batch = 1 << (int)(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(trt_logger) as builder, builder.create_network(explicit_batch) as network, trt.OnnxParser(network, trt_logger) as parser:
        # builder.max_workspace_size = 1 << 30
        builder.max_batch_size = batch_size

        if trt_engine_datatype == trt.DataType.FLOAT:
            pass
        elif trt_engine_datatype == trt.DataType.HALF:
            builder.fp16_mode = True
        else:
            raise NotImplementedError

        model = modify_onnx_model(onnx_model_path, batch_size)
        model_str = model.SerializeToString()

        parse_status = parser.parse(model_str)
        if not parse_status:
            error = parser.get_error(0)
            msg = "While parsing node number %i:\n" % error.node()
            msg += ("%s:%i In function %s:\n[%i] %s" %
                    (error.file(), error.line(), error.func(),
                     error.code(), error.desc()))
            raise RuntimeError(msg)

        if not silent:
            print("Building TensorRT engine. This may take few minutes.")

        return builder.build_cuda_engine(network)

def save_engine(engine, engine_dest_path):
    """
    Save the serialized TensorRT engine to path
    
    Parameters
    ----------
    engine : tensorrt.ICudaEngine
        TensorRT engine
    engine_dest_path : str
        path to save serialized engine
    """

    buf = engine.serialize()
    with open(engine_dest_path, 'wb') as f:
        f.write(buf)

def load_engine(trt_runtime, engine_path):
    """
    Load serialized TensorRT engine 

    Parameters
    ----------
    trt_runtime :  tensorrt.Runtime
        TensorRT runtime context
    engine_path : str
        path to load serialized engine

    Returns
    -------
    tensorrt.ICudaEngine
        Deserialized TensorRT engine
    """

    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine

def namedtupledict(typename, field_names, *args, **kwargs):  # type: (Text, Sequence[Text], *Any, **Any) -> Type[Tuple[Any, ...]]
    """Generate named tuple dict class

    Parameters
    ----------
    typename : List[str]
        Type's name
    field_names : List[str]
        List of typename for dict keys

    Returns
    -------
    namedtuple
        namedtuple class for the given data
    """    
    
    field_names_map = {n: i for i, n in enumerate(field_names)}
    # Some output names are invalid python identifier, e.g. "0"
    kwargs.setdefault(str('rename'), True)
    data = namedtuple(typename, field_names, *args, **kwargs)  # type: ignore

    def getitem(self, key):  # type: (Any, Any) -> Any
        if isinstance(key, string_types):
            key = field_names_map[key]
        return super(type(self), self).__getitem__(key)  # type: ignore
    data.__getitem__ = getitem
    return data
