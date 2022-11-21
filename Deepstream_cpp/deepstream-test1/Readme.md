# Deepstream test 1
Create sample deeptream app that load alex net in tensorrt format to classify image

## How to run

Create AlexNet in TensorRT format, convert from torch to onnx format
```
python torch_2_onnx.py
```

Convert from onnx to tensorrt format
```
python onnx_2_trt.py --verbose --model ./weights/alexnet
```