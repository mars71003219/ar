"""
분리된 포즈 파이프라인 설정 파일
"""

# 입출력 경로
input_dir = "/workspace/rtmo_gcn_pipeline/rtmo_pose_track/test_data"
output_dir = "/workspace/rtmo_gcn_pipeline/rtmo_pose_track/output"

# 모델 설정
detector_config = "/workspace/mmpose/configs/body_2d_keypoint/rtmo/body7/rtmo-m_16xb16-600e_body7-640x640.py"
detector_checkpoint = "/workspace/mmpose/checkpoints/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.pth"
device = "cuda:0"

# 포즈 추정 설정
score_thr = 0.3
nms_thr = 0.35

# 윈도우 설정
clip_len = 100
training_stride = 10

# 트래킹 설정
track_high_thresh = 0.6
track_low_thresh = 0.1
track_max_disappeared = 30
track_min_hits = 3

# 품질 설정
quality_threshold = 0.3
min_track_length = 10

# 복합점수 가중치 설정
weights = {
    'movement': 0.40,           # 움직임 강도 (가장 중요)
    'position': 0.15,           # 위치 점수 (영역별 중요도)
    'interaction': 0.30,        # 상호작용 점수 (두 번째로 중요)
    'temporal_consistency': 0.08,  # 시간적 일관성
    'persistence': 0.02         # 지속성 점수
}

# 기존 리스트 방식과의 호환성을 위한 헬퍼 함수
def get_weights_list():
    """가중치를 리스트 형태로 반환 (기존 코드 호환성)"""
    return [
        weights['movement'],
        weights['position'],
        weights['interaction'],
        weights['temporal_consistency'],
        weights['persistence']
    ]

# 데이터셋 분할 비율
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# 병렬 처리 설정
max_workers = 32  # 모델 재로딩 방지를 위해 1로 설정