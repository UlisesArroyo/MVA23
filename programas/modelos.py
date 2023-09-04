from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.models import convnext_tiny
from torchvision.models import convnext_small
#from intern_image import interImageSmallCustom, internImageSmallCAB
from torchvision.models.convnext import LayerNorm2d
from torchvision.models import resnet50
from torchvision.models import resnext50_32x4d
from torchvision.models import mobilenet_v3_large
from torchvision.models import efficientnet_b0



def convNext(n_class):

    
    #model= convnext_tiny(pretrained=True, progress=True)
    model = convnext_small(pretrained=True, progress=True)

    #print(model)

    model.named_children()
    n_inputs = None
    for name, child in model.named_children():
        if name == 'classifier':
            for sub_name, sub_child in child.named_children():
                if sub_name == '2':
                    n_inputs = sub_child.in_features
    sequential_layers = nn.Sequential(
        LayerNorm2d((768,), eps=1e-06, elementwise_affine=True),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(n_inputs, n_class, bias=True),
        nn.LogSoftmax(dim=1)
    )

    model.classifier = sequential_layers

    return model, "convnext"

def Resnet50(n_class):
    model = resnet50(pretrained=True, progress=True)

    #print(model)

    model.named_children()
    n_inputs = None
    for name, child in model.named_children():
        if name == 'fc':
            n_inputs = child.in_features

    linear_layer = nn.Sequential(
        nn.Linear(n_inputs, n_class, bias=True),
        nn.LogSoftmax(dim=1)
    )
    
    model.classifier = linear_layer

    return model, "resnet50"

def ResNeXt50(n_class):
    model = resnext50_32x4d(pretrained=True, progress=True)

    print(model)

    model.named_children()
    n_inputs = None
    for name, child in model.named_children():
        if name == 'fc':
            n_inputs = child.in_features

    linear_layer = nn.Sequential(
        nn.Linear(n_inputs, n_class, bias=True),
        nn.LogSoftmax(dim=1)
    )
    
    model.classifier = linear_layer

    return model, "resnext50"


def MobileNetV3(n_class):
    model= mobilenet_v3_large(pretrained=True, progress=True)

    print(model)

    model.named_children()
    n_inputs = None
    for name, child in model.named_children():
        if name == 'classifier':
            for sub_name, sub_child in child.named_children():
                if sub_name == '3':
                    n_inputs = sub_child.in_features
    sequential_layers = nn.Sequential(
        nn.Linear(in_features=960, out_features=1280, bias=True),
        nn.Hardswish(),
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(n_inputs, n_class, bias=True),
        nn.LogSoftmax(dim=1)
    )

    model.classifier = sequential_layers

    return model, "mobilenetv3"


def EfficientNet(n_class):
    model = efficientnet_b0(pretrained=True, progress=True)

    #print(model)

    model.named_children()
    n_inputs = None
    for name, child in model.named_children():
        if name == 'classifier':
            for sub_name, sub_child in child.named_children():
                if sub_name == '1':
                    n_inputs = sub_child.in_features
    sequential_layers = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(n_inputs, n_class, bias=True),
        nn.LogSoftmax(dim=1)
    )
    
    model.classifier = sequential_layers

    return model, "efficientnet"

def interImage(n_class):

    model = interImageSmallCustom(n_class)

    return model, "internimage"
    