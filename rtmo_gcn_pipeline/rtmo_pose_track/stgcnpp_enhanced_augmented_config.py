"""
STGCN++ 향상된 데이터 증강 설정
Dense training data (stride=10) + 적극적인 데이터 증강 적용
"""

# model settings
model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        gcn_adaptive=True,
        gcn_with_res=True,
        tcn_type='mstcn',
        graph_cfg=dict(layout='coco', strategy='spatial')),
    cls_head=dict(type='GCNHead', num_classes=2, in_channels=256))

# dataset settings
dataset_type = 'PoseDataset'
ann_file_train = '/workspace/rtmo_gcn_pipeline/rtmo_pose_track/output/dense_training/rwf2000_enhanced_sliding_train.pkl'
ann_file_val = '/workspace/rtmo_gcn_pipeline/rtmo_pose_track/output/dense_training/rwf2000_enhanced_sliding_val.pkl'
ann_file_test = '/workspace/rtmo_gcn_pipeline/rtmo_pose_track/output/sparse_inference/rwf2000_enhanced_sliding_test.pkl'

left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]

# 🚀 향상된 학습용 데이터 증강 파이프라인
train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco'),
    
    # 1. 좌우 반전 (가장 기본적이고 효과적)
    dict(type='RandomFlip', 
         flip_ratio=0.5,
         left_kp=left_kp, 
         right_kp=right_kp),
    
    # 2. 회전 변형 (자연스러운 시점 변화)
    dict(type='RandomRot',
         rot_range=[-15, 15],  # ±15도 회전
         prob=0.4),
    
    # 3. 크기 조절 (다양한 거리감)
    dict(type='RandomScale',
         scale_range=[0.85, 1.15],  # 85%-115% 크기 변화
         prob=0.5),
    
    # 4. 전단 변형 (카메라 각도 변화 시뮬레이션)
    dict(type='RandomShear',
         shear_range=[-10, 10],  # ±10도 전단
         prob=0.3),
    
    # 5. 노이즈 추가 (센서 노이즈, 가려짐 시뮬레이션)
    dict(type='RandomJitter',
         noise_std=0.05,  # 키포인트 위치에 가우시안 노이즈
         prob=0.4),
    
    # 6. 키포인트 가려짐 (일부 관절이 보이지 않는 상황)
    dict(type='RandomOcclusion',
         occlusion_ratio=0.1,  # 10% 키포인트 가려짐
         prob=0.3),
    
    # 7. 시간적 일관성 변형 (프레임 순서 약간 변경)
    dict(type='RandomTemporalSubSample',
         temporal_jitter_ratio=0.05,  # 5% 시간적 변동
         prob=0.2),
    
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

# 🎯 검증/추론용 파이프라인 (증강 없음)
val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

data = dict(
    videos_per_gpu=32,  # 배치 크기 (GPU 메모리에 따라 조정)
    workers_per_gpu=4,
    test_dataloader=dict(videos_per_gpu=1),
    
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix='',
        pipeline=train_pipeline),  # 증강 파이프라인 적용
    
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix='',
        pipeline=val_pipeline),   # 증강 없음
    
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix='',
        pipeline=test_pipeline))  # 추론용 sparse 데이터 + 증강 없음

# optimizer
optimizer = dict(
    type='AdamW', 
    lr=0.001,         # 낮은 학습률 (AdamW에 적합)
    weight_decay=0.01,
    betas=(0.9, 0.999))

optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.0001,    # 최소 학습률도 낮게 조정
    warmup='linear',
    warmup_iters=1000,     # 더 긴 워밍업 (AdamW에 유리)
    warmup_ratio=0.1)

total_epochs = 80          # 에폭 수 (더 많은 데이터로 더 오래 학습)

checkpoint_config = dict(interval=5)
evaluation = dict(
    interval=5, 
    metrics=['top_k_accuracy', 'mean_class_accuracy'],
    topk=(1, 2))

log_config = dict(
    interval=50,           # 로그 출력 간격 (더 자주)
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/stgcnpp_enhanced_augmented_fight_detection'
load_from = None
resume_from = None
workflow = [('train', 1)]

# 🎯 핵심 개선사항 요약:
# 1. Dense training data: 16,371 segments (2.8x more)
# 2. 7가지 데이터 증강: flip, rotation, scale, shear, jitter, occlusion, temporal
# 3. 학습/추론 데이터 분리: dense training, sparse inference
# 4. 향상된 하이퍼파라미터: 높은 학습률, 긴 학습, 코사인 스케줄링