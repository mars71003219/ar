# visualizer_config.py

# --- 경로 설정 ---
# 원본 비디오가 있는 디렉토리
input_dir = "/aivanas/raw/surveillance/action/violence/action_recognition/data/RWF-2001"
# 추론 결과 .pkl 파일이 있는 디렉토리
output_dir = "/workspace/rtmo_gcn_pipeline/rtmo_pose_track/output/test_visualizer"
# pkl 파일을 검색할 output_dir 내의 하위 디렉토리
PKL_SEARCH_SUBDIR = "windows"

# --- 시각화 설정 ---
num_persons = 4  # 시각화할 최대 인원 수
confidence_threshold = 0.3  # 관절 포인트를 표시할 최소 신뢰도
verbose = True  # 처리 과정 로그 출력 여부

# 지원하는 비디오 확장자
supported_video_extensions = ["mp4", "avi", "mov", "mkv"]

# --- 색상 설정 (BGR) ---
person_colors = [
    (0, 255, 0),   # Green
    (0, 0, 255),   # Red
    (255, 0, 0),   # Blue
    (0, 255, 255), # Cyan
    (255, 0, 255)  # Magenta
]
overlap_color = (0, 165, 255)  # Orange for overlap
fight_color = (0, 0, 180)      # Dark Red for fight window
nonfight_color = (180, 0, 0)   # Dark Blue for non-fight window
final_fight_color = (0, 0, 255) # Bright Red for final fight detection
final_nonfight_color = (255, 0, 0) # Bright Blue for final non-fight

# --- UI 텍스트 및 레이아웃 설정 ---
font_scale = 0.3
font_thickness = 1
title_font_scale = 0.3
title_font_thickness = 1
window_info_x = 10
window_info_y_start = 20
window_info_y_step = 15
frame_info_margin = 5
final_result_margin = 5

# --- 최종 판정 로직 설정 ---
consecutive_threshold = 3 # Fight로 판정하기 위한 연속적인 Fight 윈도우 최소 개수

# --- 비디오 저장 설정 ---
output_video_codec = 'mp4v'  # 'mp4v' for .mp4, 'XVID' for .avi

# --- 스켈레톤 연결 정보 (COCO 17 keypoints) ---
skeleton_connections = [
    (1, 2), (1, 3), (2, 4), (3, 5), (6, 7), (6, 8), (7, 9), (8, 10),
    (12, 13), (12, 14), (13, 15), (14, 16), (6, 12), (7, 13)
]

# 오버레이 비디오 저장 설정
SAVE_OVERLAY_VIDEO = True  # True로 설정하면 오버레이 비디오를 저장합니다.
OVERLAY_SUB_DIR = 'overlay' # output_dir 하위에 생성될 서브디렉토리 이름
