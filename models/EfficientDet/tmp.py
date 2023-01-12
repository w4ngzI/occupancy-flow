from backbone import EfficientDetBackbone
import numpy as np
import torch
import torch.nn as nn
anchors_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchors_scales = [4., 4., 4., 4., 4., 4., 4., 5., 4.]
model = EfficientDetBackbone(num_classes=3, compound_coef=5)

img = torch.ones((2, 64, 512, 512))

features = model(img)

for i in range(len(features)):
    print(features[i].shape)

upsample = nn.ConvTranspose2d(288, 288, 3, stride=2, padding=1)
a = upsample(features[0], output_size = (2, 288, 128, 128))
print(a.shape)

conv = nn.Conv2d(288, 720, 3, stride=1, padding=1)
b = conv(a)
print(b.shape)