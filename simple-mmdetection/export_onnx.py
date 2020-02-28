from models.detectors import create_detector
import torch
import torchvision
import cv2
import numpy as np
from matplotlib import pyplot as plt

import onnx
import torch.onnx

# pretrained checkpoint.
model = create_detector('freeanchor_retinanet_r50_fpn',
                        number_of_classes=2,
                        pretrained='retinanet_free_anchor_r50_fpn_nmspre.pth')

print(model)
img_height = 480
img_width = 800
output_model_path = 'retinanet_free_anchor_r50_fpn_nmspre.onnx'

x = torch.randn(1, 3, img_height, img_width, requires_grad=True)

onnx_model  = torch.onnx.export(model,
    x,
    output_model_path,
    export_params=True,
    )
print('Export onnx done.')

# with pytorch 1.3, model can be easily quantized (better CPU performance, smaller footprint).
#model = torch.quantization.quantize_dynamic(retina, dtype=torch.qint8)

onnx_model = onnx.load(output_model_path)
import caffe2.python.onnx.backend as onnx_caffe2_backend
rep = onnx_caffe2_backend.prepare(onnx_model, device="CPU")
print('Caffe2 backend done.')
outputs = rep.run(np.random.randn(1, 3, img_height, img_width).astype(np.float32))
print(outputs)
print('Done.')
