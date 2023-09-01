
import torch
import numpy as np
import torchvision
from torchvision import transforms
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

import onnxruntime

from time import perf_counter
from utils import measure_time, measure_memory_usage, profile

data_transforms = {
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

class Image_Folder(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.im_names = next(os.walk(root_dir))[2]

    def __len__(self):
        return 200
        return len(self.im_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.im_names[idx])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image, torch.from_numpy(np.array([1, ]))
    

    
    
data_dir = '../../Datasets/mosquito_alert_dataset/final/'
image_datasets = {x: Image_Folder(data_dir, data_transforms[x])
                  for x in ['val',]}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                             shuffle=False, num_workers=0)
              for x in ['val', ]}
dataset_sizes = {x: len(image_datasets[x]) for x in ['val', ]}
class_names = ["mosquito", ]

device = torch.device("cpu")

model_conv = torchvision.models.resnet18(weights='IMAGENET1K_V1')
for param in model_conv.parameters():
    param.requires_grad = False
    
# Input to the model
x = torch.randn(1, 3, 224, 224, requires_grad=True)
torch_out = model_conv(x)

# Export the model
torch.onnx.export(model_conv,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "resnet18.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})

ort_session = onnxruntime.InferenceSession("resnet18.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

@measure_time
def infer_time():
    infer_times = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            time_start = perf_counter()
            inputs = inputs.to(device)
            labels = labels.to(device)

            # compute ONNX Runtime output prediction
            ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(inputs)}
            ort_outs = ort_session.run(None, ort_inputs)

            time_stop = perf_counter()
            infer_times.append(time_stop-time_start)
    print("max: ", max(infer_times))
    print("average: ", sum(infer_times)/len(infer_times))
            
@profile
def infer_memory():
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # compute ONNX Runtime output prediction
            ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(inputs)}
            ort_outs = ort_session.run(None, ort_inputs)

infer_time()
infer_memory()