import os
import torch
import torchvision

def fuse_conv_and_bn(conv, bn):
    # Initialization
    fused_conv = torch.nn.Conv2d(conv.in_channels, conv.out_channels,
                                kernel_size=conv.kernel_size, 
                                stride=conv.stride,
                                padding=conv.padding,
                                bias=True)
    # prepare filter
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fused_conv.weight.copy_(torch.mm(w_bn, w_conv).view(fused_conv.weight.size()))
    
    # prepare spatial bias
    if conv.bias is not None:
        b_conv = conv.bias
    else:
        b_conv = torch.zeros(conv.weight.size(0))
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fused_conv.bias.copy_(b_conv + b_bn)
    return fused_conv

   
torch.set_grad_enabled(False)

x = torch.randn(16, 3, 256, 256)
resnet18 = torchvision.models.resnet18(pretrained=True)
resnet18.eval()

net = torch.nn.Sequential(
    resnet18.conv1,
    resnet18.bn1    
)

y1 = net.forward(x)
fused_conv = fuse_conv_and_bn(net[0], net[1])
y2 = fused_conv.forward(x)
d = (y1 - y2).norm().div(y1.norm()).item()
print("error: %.8f" % d)
   
