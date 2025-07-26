# 이 파일은 모델의 전체 구조(backbone + cls_head)를 정의한다. (수정된 버전)
_base_ = 'stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py'


# 1. 모델 설정 수정
model = dict(
    cls_head=dict(
        type='mmaction.GCNHead',  # <- 수정
        num_classes=2,  # 분류할 클래스 수를 2개(fight/non-fight)로 변경
        in_channels=256,
        loss_cls=dict(type='mmaction.CrossEntropyLoss'))) # <- 수정

# 2. 데이터셋 설정
dataset_type = 'mmaction.PoseDataset' # <- 수정
# RWF-2000 데이터셋의 루트 경로를 지정한다.
# 실제 데이터셋의 비디오(.mp4) 파일이 있는 상위 폴더 경로를 입력해야 한다.
data_root = '/workspace/rtmo_gcn_pipeline/rtmo_pose_track/output/RWF-2000/'
# RTMO로 추출한 RWF-2000 용 어노테이션 파일 경로
# 실제 생성한 .pkl 파일 이름으로 수정해야 한다.
ann_file_train = '/workspace/rtmo_gcn_pipeline/rtmo_pose_track/output/rwf2000_train.pkl'
ann_file_val = '/workspace/rtmo_gcn_pipeline/rtmo_pose_track/output/rwf2000_val.pkl'

train_pipeline = [
    dict(type='mmaction.PreNormalize2D'), # <- 수정
    # 사전학습 시 사용한 'bone' 특징을 동일하게 사용해 일관성 유지
    dict(type='mmaction.GenSkeFeat', dataset='coco', feats=['b']), # <- 수정
    dict(type='mmaction.UniformSampleFrames', clip_len=100), # <- 수정
    dict(type='mmaction.PoseDecode'), # <- 수정
    dict(type='mmaction.FormatGCNInput', num_person=2), # RWF-2000도 주로 2명 간의 상호작용이므로 2로 유지 # <- 수정
    dict(type='mmaction.PackActionInputs') # <- 수정
]
val_pipeline = [
    dict(type='mmaction.PreNormalize2D'), # <- 수정
    dict(type='mmaction.GenSkeFeat', dataset='coco', feats=['b']), # <- 수정
    dict(
        type='mmaction.UniformSampleFrames', clip_len=100, num_clips=1, test_mode=True), # <- 수정
    dict(type='mmaction.PoseDecode'), # <- 수정
    dict(type='mmaction.FormatGCNInput', num_person=2), # <- 수정
    dict(type='mmaction.PackActionInputs') # <- 수정
]
test_pipeline = [
    dict(type='mmaction.PreNormalize2D'), # <- 수정
    dict(type='mmaction.GenSkeFeat', dataset='coco', feats=['b']), # <- 수정
    dict(
        type='mmaction.UniformSampleFrames', clip_len=100, num_clips=10, test_mode=True), # <- 수정
    dict(type='mmaction.PoseDecode'), # <- 수정
    dict(type='mmaction.FormatGCNInput', num_person=2), # <- 수정
    dict(type='mmaction.PackActionInputs') # <- 수정
]

train_dataloader = dict(
    _delete_=True,
    batch_size=16, # GPU 메모리에 따라 조절 (8, 16, 32 등)
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='mmaction.DefaultSampler', shuffle=True), # <- 수정
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        # data_prefix=dict(img=data_root), # 원본 비디오 경로 prefix
        pipeline=train_pipeline,
        num_classes=2)) # 데이터셋에도 클래스 수 명시

val_dataloader = dict(
    _delete_=True,
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='mmaction.DefaultSampler', shuffle=False), # <- 수정
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        # data_prefix=dict(img=data_root),
        pipeline=val_pipeline,
        num_classes=2,
        test_mode=True))

test_dataloader = val_dataloader # 검증과 테스트에 동일한 데이터로더 설정 사용

# 3. 학습 전략 (Optimizer, Scheduler) 설정
optim_wrapper = dict(
    _delete_=True,
    type='mmaction.OptimWrapper', # <- 수정
    optimizer=dict(type='mmaction.AdamW', lr=5e-4, weight_decay=0.0005), # 기본 학습률을 낮게 설정 # <- 수정
    paramwise_cfg=dict(
        custom_keys={
            # 모델의 몸통(backbone)은 훨씬 낮은 학습률로 미세 조정
            'backbone': dict(lr_mult=0.1)
        }))

param_scheduler = [
    dict(
        type='mmaction.CosineAnnealingLR', # 코사인 스케줄러로 점진적 학습률 감소 # <- 수정
        by_epoch=True,
        begin=0,
        T_max=30, # 전체 학습 에폭 수
        eta_min_ratio=0.1) # 최소 학습률 비율
]

# 4. 학습, 검증, 테스트 루프 설정
train_cfg = dict(
    max_epochs=30, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop') # MMEngine 고유 기능이므로 수정하지 않음
test_cfg = dict(type='TestLoop') # MMEngine 고유 기능이므로 수정하지 않음

# 5. 사전 학습 가중치 로드
# 다운로드 받은 NTU-60 학습 완료 모델의 .pth 파일 경로를 지정해야 한다.
load_from = '/workspace/mmaction2/checkpoints/stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d_20221228-cd11a691.pth'