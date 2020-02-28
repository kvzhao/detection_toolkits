import torch
import torchvision
import cv2
import numpy as np

import onnx
import torch.onnx

from models.detectors import create_detector


def main(args):
    pretrained_model_path = args.model_path
    # pretrained checkpoint.
    model = create_detector(args.model_type,
                            number_of_classes=2,
                            pretrained=pretrained_model_path)

    print(model)
    # Not sure how to handle input size in openvino
    img_height = args.input_image_height
    img_width = args.input_image_width
    output_model_path = args.output_path

    dummy_input = torch.randn(1, 3, img_height, img_width, requires_grad=True)

    onnx_model = torch.onnx.export(model,
                                    dummy_input,
                                    output_model_path,
                                    export_params=True,)
    print('Export onnx done.')

    # Torch version 1.0.1
    # The following codes for check
    onnx_model = onnx.load(output_model_path)
    import caffe2.python.onnx.backend as onnx_caffe2_backend
    rep = onnx_caffe2_backend.prepare(onnx_model, device="CPU")
    print('Caffe2 backend done.')
    outputs = rep.run(np.random.randn(1, 3, img_height, img_width).astype(np.float32))
    print(outputs)
    print('Done.')
    # TODO: convert to openvino


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='ONNX Model Converter')
    parser.add_argument('-m', '--model_path', default=None, type=str,
                        help='Path to the pytorch .pth')
    parser.add_argument('-o', '--output_path', default=None, type=str,
                        help='Path to the exported .onnx')
    parser.add_argument('-mt', '--model_type', default='freeanchor_retinanet_r50_fpn', type=str,
                        help='Type: freeanchor_retinanet_r50_fpn, freeanchor_retinanet_r50_fpn_bn')
    parser.add_argument('--input_image_width', default=800, type=int)
    parser.add_argument('--input_image_height', default=480, type=int)

    args = parser.parse_args()
    main(args)
