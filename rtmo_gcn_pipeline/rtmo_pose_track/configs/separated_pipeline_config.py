"""
분리된 포즈 파이프라인 설정 파일
"""

# 입출력 경로
input_dir = "/workspace/rtmo_gcn_pipeline/rtmo_pose_track/raw_videos"
output_dir = "/workspace/rtmo_gcn_pipeline/rtmo_pose_track/output/hanbit-sep"

# 모델 설정
detector_config = "/workspace/mmpose/configs/body_2d_keypoint/rtmo/body7/rtmo-m_16xb16-600e_body7-640x640.py"
detector_checkpoint = "/workspace/mmpose/checkpoints/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.pth"
device = "cuda:0"

# 포즈 추정 설정
score_thr = 0.4
nms_thr = 0.35

# 윈도우 설정
clip_len = 100
training_stride = 10

# 트래킹 설정
track_high_thresh = 0.8
track_low_thresh = 0.1
track_max_disappeared = 5
track_min_hits = 2

# 품질 설정
quality_threshold = 0.15
min_track_length = 5

# 복합점수 가중치 설정
movement_weight = 0.40           # 움직임 강도 (가장 중요)
position_weight = 0.15           # 위치 점수 (영역별 중요도)
interaction_weight = 0.30        # 상호작용 점수 (두 번째로 중요)
temporal_consistency_weight = 0.08  # 시간적 일관성
persistence_weight = 0.02         # 지속성 점수

# 데이터셋 분할 비율
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# 병렬 처리 설정
max_workers = 32  # 모델 재로딩 방지를 위해 1로 설정