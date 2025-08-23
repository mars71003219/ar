# Copyright (c) OpenMMLab. All rights reserved.
# This script serves the sole purpose of converting skeleton-based graph
# in MMAction2 to ONNX files. Please note that attempting to convert other
# models using this script may not yield successful results.
import argparse

import numpy as np
import onnxruntime
import torch
import torch.nn as nn
from mmengine import Config
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmengine.structures import LabelData

# mmaction2 imports
# 'build_dataloader' import 라인을 제거했습니다.
from mmaction.registry import MODELS, DATASETS
from mmengine.dataset import Compose

# ActionDataSample은 mmaction.structures에 있습니다.
from mmaction.structures import ActionDataSample

def parse_args():
    parser = argparse.ArgumentParser(description='Get model flops and params')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--num_frames', type=int, default=150, help='number of input frames.')
    parser.add_argument(
        '--num_person', type=int, default=2, help='number of maximum person.')
    parser.add_argument(
        '--num_joints',
        type=int,
        default=0,
        help='number of joints. If not given, will use default settings from'
        'the config file')
    parser.add_argument(
        '--device', type=str, default='cpu', help='CPU/CUDA device option')
    parser.add_argument(
        '--output_file',
        type=str,
        default='stgcn.onnx',
        help='file name of the output onnx file')
    args = parser.parse_args()
    return args


class AvgPool2d(nn.Module):

    def forward(self, x):
        return x.mean(dim=(-1, -2), keepdims=True)


class MaxPool2d(nn.Module):

    def forward(self, x):
        x = x.max(dim=-1, keepdim=True)[0]
        x = x.max(dim=-2, keepdim=True)[0]
        return x


class GCNNet(nn.Module):

    def __init__(self, base_model):
        super(GCNNet, self).__init__()
        self.backbone = base_model.backbone
        self.head = base_model.cls_head

        if hasattr(self.head, 'pool'):
            pool = self.head.pool
            if isinstance(pool, nn.AdaptiveAvgPool2d):
                assert pool.output_size == 1
                self.head.pool = AvgPool2d()
            elif isinstance(pool, nn.AdaptiveMaxPool2d):
                assert pool.output_size == 1
                self.head.pool = MaxPool2d()

    def forward(self, input_tensor):
        feat = self.backbone(input_tensor)
        cls_score = self.head(feat)
        return cls_score


def softmax(x):
    x = np.exp(x - x.max())
    return x / x.sum()


def main():
    args = parse_args()
    config = Config.fromfile(args.config)
    init_default_scope(config.get('default_scope', 'mmaction'))

    if config.model.type != 'RecognizerGCN':
        print(
            'This script serves the sole purpose of converting skeleton-based '
            'graph in MMAction2 to ONNX files. Please note that attempting to '
            'convert other models using this script may not yield successful '
            'results.\n\n')
    # === 수정 시작: 실제 데이터 로더에서 입력 텐서 가져오기 ===
    # 1. 데이터셋 파이프라인 생성
    # val_pipeline 또는 test_pipeline 사용 가능
    pipeline = Compose(config.val_pipeline) 
    
    # 2. 데이터셋 빌드 (실제 데이터 파일이 필요)
    # config 파일의 val_dataloader 설정을 기반으로 데이터셋 생성
    val_dataset_cfg = config.val_dataloader.dataset
    dataset = DATASETS.build(val_dataset_cfg)

    # 3. 데이터셋에서 하나의 샘플 가져오기
    # 첫 번째 데이터를 예시로 사용
    data_sample_from_dataset = dataset[0] 
    
    # 4. 파이프라인을 통해 전처리 수행
    processed_sample = pipeline(data_sample_from_dataset)
    
    # 5. 실제 모델에 들어갈 입력 텐서 추출
    # Dataloader가 collate하는 것과 유사하게 수동으로 batch 차원 추가
    # processed_sample['inputs']는 리스트 형태이므로 첫 번째 요소를 사용
    input_tensor = processed_sample['inputs'][0].unsqueeze(0).to(args.device) 
    # 최종 input_tensor shape: [1, num_person, num_frames, num_joints, C]
    # === 수정 끝 ===
    
    base_model = MODELS.build(config.model)
    load_checkpoint(base_model, args.checkpoint, map_location='cpu')
    base_model.to(args.device)
    base_model.eval()

    # === 수정: 입력 Shape 통일 ===
    # PyTorch 모델 추론 시 .unsqueeze(0) 제거
    data_sample = ActionDataSample()
    # base_model은 [inputs] 리스트와 data_samples 리스트를 받음
    base_output_raw = base_model([input_tensor], data_samples=[data_sample], mode='predict')[0]
    base_output = base_output_raw.pred_score.detach().cpu().numpy()

    model = GCNNet(base_model).to(args.device)
    model.eval()

    torch.onnx.export(
        model, (input_tensor),
        args.output_file,
        input_names=['input_tensor'],
        output_names=['cls_score'],
        export_params=True,
        do_constant_folding=True,
        verbose=False,
        opset_version=12,
        dynamic_axes={
            'input_tensor': {
                0: 'batch_size',
                1: 'num_person',
                2: 'num_frames'
            },
            'cls_score': {
                0: 'batch_size'
            }
        })

    print(f'Successfully export the onnx file to {args.output_file}')

    # Test exported file
    session = onnxruntime.InferenceSession(args.output_file)
    input_feed = {'input_tensor': input_tensor.cpu().data.numpy()}
    outputs = session.run(['cls_score'], input_feed=input_feed)
    # === 수정: 검증 로직 통일 (softmax 없이 로짓 값으로 비교) ===
    onnx_output = outputs[0]

    # 비교를 위해 shape 맞추기 (base_output: (1, num_classes), onnx_output: (1, num_classes))
    print(f'PyTorch Logits: {base_output}')
    print(f'ONNX Logits: {onnx_output}')

    diff = abs(base_output - onnx_output).max()
    print(f'Max absolute difference between PyTorch and ONNX logits: {diff:.6f}')
    if diff < 1e-5:
        print('The output difference is smaller than 1e-5.')
    else:
        print('The output difference is larger than 1e-5. Please check the conversion process.')



if __name__ == '__main__':
    main()
