# Compare CPU Inference
This folder is to compare inference time, speed and memory consumption on CPU. 

Framework: Pytorch, OpenVINO, ONNX-runtime

## Results
WARNING: result get with 200 images

Model: Resnet 18
|Framwork | Infer speed max (second/im) | Infer speed average (second/im) | Mem consumptions (b/MB)
|-----|--------|--------|--------|
|Pytorch   |0.4667382590705529      |0.04263776458217763      | 336,035,840  / 300      |
|Onnx-cpu  |0.12485788099002093       | 0.024308491967385635       | 590,581,760 / 590      |
|OpenVINO  |0.10971567500382662      |0.02527220929856412      |854,892,544 / 800     |


Model: YOLOX
Do later

## Installation
```
conda create -n compare_cpu python=3.9
```

```
conda activate compare_cpu
```

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

```
pip install -r requirements.txt
```