import tensorrt as trt
import torch

def trt_version():
    return trt.__version__


def torch_version():
    return torch.__version__

def torch_dtype_from_trt(dtype):
    if dtype == trt.int8:
        return torch.int8
    elif trt_version() >= '7.0' and dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError("%s is not supported by torch" % dtype)


def torch_device_from_trt(device):
    if device == trt.TensorLocation.DEVICE:
        return torch.device("cuda")
    elif device == trt.TensorLocation.HOST:
        return torch.device("cpu")
    else:
        return TypeError("%s is not supported by torch" % device)

class TRTModule(torch.nn.Module):
    def __init__(self, engine=None, input_names=None, output_names=None):
        super(TRTModule, self).__init__()
        self.engine = engine
        if self.engine is not None:
            # engine创建执行context
            self.context = self.engine.create_execution_context()

        self.input_names = input_names
        self.output_names = output_names

    def forward(self, *inputs):
        batch_size = inputs[0].shape[0]
        bindings = [None] * (len(self.input_names) + len(self.output_names))

        for i, input_name in enumerate(self.input_names):
            idx = self.engine.get_binding_index(input_name)
            # 设定shape
            self.context.set_binding_shape(idx, tuple(inputs[i].shape))
            bindings[idx] = inputs[i].contiguous().data_ptr()

        # create output tensors
        outputs = [None] * len(self.output_names)
        for i, output_name in enumerate(self.output_names):
            idx = self.engine.get_binding_index(output_name)
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            shape = tuple(self.context.get_binding_shape(idx))
            device = torch_device_from_trt(self.engine.get_location(idx))
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[i] = output
            bindings[idx] = output.data_ptr()

        self.context.execute_async_v2(bindings,
                                      torch.cuda.current_stream().cuda_stream)

        outputs = tuple(outputs)
        if len(outputs) == 1:
            outputs = outputs[0]

        return outputs

class RunTensorRT(object):
    def __init__(self, modelp = r"D:\doc\Work\jgtd\models\yolov5_20220429\best.trt"):
        self.logger = trt.Logger(trt.Logger.INFO)
        self.model_path = modelp
        EXPLICIT_BATCH = []
        self.model_all_names = []
        with open(self.model_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            for idx in range(self.engine.num_bindings):
                is_input = self.engine.binding_is_input(idx)
                name = self.engine.get_binding_name(idx)
                op_type = self.engine.get_binding_dtype(idx)
                self.model_all_names.append(name)
                shape = self.engine.get_binding_shape(idx)

                print('input id:', idx, '   is input: ', is_input, '  binding name:', name, '  shape:', shape, 'type: ',
                  op_type)
        self.trt_model = TRTModule(self.engine, ["input"], ["output"])


    def run(self, inp):
        result_trt = self.trt_model(inp)
        return result_trt

if __name__ == "__main__":
    trt = RunTensorRT()
    print(trt)

