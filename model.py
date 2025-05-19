# The main model used in this project. pretrained ResNet and GoogLeNet + LORA in selected layers

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, googlenet, ResNet18_Weights, GoogLeNet_Weights

class LoRAConv2d(nn.Module):
    def __init__(self, base_layer, rank=4, scale=1.0):
        super().__init__()
        if hasattr(base_layer, 'conv') and isinstance(base_layer.conv, nn.Conv2d):
            base_conv = base_layer.conv
        elif isinstance(base_layer, nn.Conv2d):
            base_conv = base_layer
        else:
            raise ValueError(f"Unsupported conv layer type: {type(base_layer)}")

        self.base_conv = base_conv
        self.base_conv.requires_grad_(False)
        in_channels = base_conv.in_channels
        out_channels = base_conv.out_channels
        self.rank = rank
        self.scale = scale
        
        # trainable convolution of LoRA which are trained
        self.lora_down = nn.Conv2d(in_channels, rank, kernel_size=1, stride=base_conv.stride, padding=0, bias=False)
        self.lora_up = nn.Conv2d(rank, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=0.01)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        with torch.no_grad():
            base_out = self.base_conv(x)
        lora_residual = self.lora_up(self.lora_down(x)) * self.scale
        return base_out + lora_residual


class CombinedModel(nn.Module):
    def __init__(self, num_classes=10, freeze=True, lora_rank=4, lora_scale=1.0):
        super().__init__()
        
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.conv1 = LoRAConv2d(self.resnet.conv1, rank=lora_rank, scale=lora_scale)
        self.resnet.layer4[1].conv1 = LoRAConv2d(self.resnet.layer4[1].conv1, rank=lora_rank, scale=lora_scale)
        self.resnet_features = nn.Sequential(*list(self.resnet.children())[:-1])

        self.googlenet = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
        self.googlenet.conv1.conv = LoRAConv2d(self.googlenet.conv1, rank=lora_rank, scale=lora_scale)
        self.googlenet.inception5b.branch1 = LoRAConv2d(self.googlenet.inception5b.branch1, rank=lora_rank, scale=lora_scale)

        # here are the layers of googlenet except the last layer
        self.googlenet_features = nn.Sequential(
            self.googlenet.conv1,
            self.googlenet.maxpool1,
            self.googlenet.conv2,
            self.googlenet.conv3,
            self.googlenet.maxpool2,
            self.googlenet.inception3a,
            self.googlenet.inception3b,
            self.googlenet.maxpool3,
            self.googlenet.inception4a,
            self.googlenet.inception4b,
            self.googlenet.inception4c,
            self.googlenet.inception4d,
            self.googlenet.inception4e,
            self.googlenet.maxpool4,
            self.googlenet.inception5a,
            self.googlenet.inception5b,
        )

        # all the layers of Resnet and Googlenet are frozen
        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False
            for param in self.googlenet.parameters():
                param.requires_grad = False
                
        # the LORA is addedfor the first convolution of both the networks, and the 4th layer and 5th inception layer of resnet and googlenet
        # respectively and these layers are trained
        for module in [self.resnet.conv1, self.resnet.layer4[1].conv1,
                       self.googlenet.conv1.conv, self.googlenet.inception5b.branch1]:
            for param in module.lora_down.parameters():
                param.requires_grad = True
            for param in module.lora_up.parameters():
                param.requires_grad = True
                
        # the flattened output from both the networks are projected to same dimensions
        self.resnet_proj = nn.Linear(512, 512)
        self.googlenet_proj = nn.Linear(1024, 512)

        # trainable classification module
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # The input passes through all ResNet and googlenet Layers except the last
        x1 = self.resnet_features(x)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.resnet_proj(x1)

        x2 = self.googlenet_features(x)
        x2 = F.adaptive_avg_pool2d(x2, (1, 1)).view(x2.size(0), -1)
        x2 = self.googlenet_proj(x2)

        x = torch.cat([x1, x2], dim=1)
        return self.fc(x)
