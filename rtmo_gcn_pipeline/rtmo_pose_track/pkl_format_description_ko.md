
# PKL 파일 포맷 설명

이 문서는 `_enhanced_stgcn_annotation.pkl` 파일의 키-값 구조를 설명합니다.

```
{
    # (정수) 영상에서 감지되고 추적된 총 인원 수.
    'total_persons': <int>,

    # (딕셔너리) 영상에 대한 메타데이터를 포함합니다.
    'video_info': {
        # (문자열) 비디오 프레임이 저장된 디렉토리 이름.
        'frame_dir': <str>,
        # (정수) 비디오의 총 프레임 수.
        'total_frames': <int>,
        # (튜플) 처리에 사용된 이미지의 형태 (일반적으로 높이, 너비).
        'img_shape': (height, width),
        # (튜플) 비디오의 원본 해상도 (높이, 너비).
        'original_shape': (height, width),
        # (정수) 비디오 클립의 레이블. 이 컨텍스트에서는 'Fight'의 경우 1, 'NonFight'의 경우 0입니다.
        'label': <int>
    },

    # (딕셔너리) 추적된 각 사람에 대한 자세한 정보를 포함합니다.
    # 키는 'person_00', 'person_01' 등입니다.
    'persons': {
        'person_XX': {
            # (정수) 추적된 사람에게 할당된 고유 ID.
            'track_id': <int>,
            # (실수) 이 사람의 관련성이나 중요성을 나타내는 종합 점수.
            'composite_score': <float>,
            # (딕셔너리) 종합 점수를 구성하는 다양한 요소의 분석.
            'score_breakdown': {
                'movement': <float>,          # 사람의 움직임과 관련된 점수.
                'position': <float>,          # 프레임에서 사람의 위치에 기반한 점수.
                'interaction': <float>,       # 다른 사람과의 상호작용을 나타내는 점수.
                'temporal_consistency': <float>, # 시간에 따른 추적 일관성 점수.
                'persistence': <float>        # 사람이 추적된 시간의 길이에 대한 점수.
            },
            # (딕셔너리) 프레임 영역별 위치 점수 분석.
            'region_breakdown': {
                'top_left': <float>,
                'top_right': <float>,
                'bottom_left': <float>,
                'bottom_right': <float>,
                'center_overlap': <float>
            },
            # (실수) 이 사람에 대한 추적의 품질.
            'track_quality': <float>,
            # (정수) 종합 점수에 따른 사람의 순위.
            'rank': <int>,
            # (딕셔너리) 사람에 대한 핵심 주석 데이터.
            'annotation': {
                # (Numpy 배열) 각 프레임의 키포인트 데이터.
                # 형태: (프레임 수, 키포인트 수, 2) - (x, y) 좌표.
                'keypoint': <np.array>,
                # (Numpy 배열) 각 해당 키포인트의 신뢰도 점수.
                # 형태: (프레임 수, 키포인트 수)
                'keypoint_score': <np.array>,
                # (정수) 이 사람에 대해 감지된 키포인트 수 (예: COCO의 경우 17).
                'num_keypoints': <int>,
                # (정수) 부모 track_id와 동일한 고유 추적 ID.
                'track_id': <int>
            }
        },
        # ... 더 많은 사람 정보
    },

    # (딕셔너리) 구성 요소로부터 composite_score를 계산하는 데 사용된 가중치.
    'score_weights': {
        'movement_intensity': <float>,
        'position_5region': <float>,
        'interaction': <float>,
        'temporal_consistency': <float>,
        'persistence': <float>
    },

    # (실수) 트랙 필터링에 사용된 품질 임계값.
    'quality_threshold': <float>,
    # (정수) 트랙이 포함되기 위해 존재해야 하는 최소 프레임 수.
    'min_track_length': <int>
}
```
