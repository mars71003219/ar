"""
기본 사용법 예제 (참고용)

⚠️  더 이상 이 파일을 직접 실행하지 마세요!
   대신 통합 메인 실행기를 사용하세요:
   
   python recognizer/main.py --mode inference --input video.mp4
   
   자세한 사용법은 USAGE.md를 참고하세요.

이 파일은 기본적인 모듈 사용법 참고용으로만 유지됩니다.
"""

import sys
from pathlib import Path

# ===== 이 파일은 참고용입니다. 실제로는 main.py를 사용하세요! =====
sys.path.insert(0, str(Path(__file__).parent.parent))

from recognizer import factory, initialize_factory
from recognizer.utils.data_structure import (
    PoseEstimationConfig, TrackingConfig, ScoringConfig, ActionClassificationConfig
)
from recognizer.pipelines.unified_pipeline import UnifiedPipeline, PipelineConfig
from recognizer.pipelines.annotation_pipeline import AnnotationPipeline, AnnotationConfig
from recognizer.visualization.pose_visualizer import PoseVisualizer
from recognizer.visualization.result_visualizer import ResultVisualizer


def basic_pose_estimation_example():
    """기본 포즈 추정 예제"""
    print("=== 기본 포즈 추정 예제 ===")
    
    # 설정 생성
    pose_config = PoseEstimationConfig(
        model_name='rtmo',
        config_file='path/to/rtmo_config.py',
        model_path='path/to/rtmo_checkpoint.pth',
        device='cuda:0'
    )
    
    # 포즈 추정기 생성
    pose_estimator = factory.create_pose_estimator('rtmo', pose_config.__dict__)
    
    if pose_estimator.initialize_model():
        print("포즈 추정기 초기화 성공!")
        
        # 비디오에서 포즈 추정
        # poses = pose_estimator.extract_video_poses('path/to/video.mp4')
        # print(f"추출된 프레임 수: {len(poses)}")
        
        # 정리
        pose_estimator.cleanup()
    else:
        print("포즈 추정기 초기화 실패!")


def basic_tracking_example():
    """기본 트래킹 예제"""
    print("\n=== 기본 트래킹 예제 ===")
    
    # 설정 생성
    tracking_config = TrackingConfig(
        tracker_name='bytetrack',
        frame_rate=30,
        track_high_thresh=0.6,
        track_low_thresh=0.1,
        new_track_thresh=0.7
    )
    
    # 트래커 생성
    tracker = factory.create_tracker('bytetrack', tracking_config.__dict__)
    
    if tracker.initialize_tracker():
        print("트래커 초기화 성공!")
        
        # 포즈 데이터에 트래킹 적용 (예제용 가상 데이터)
        # tracked_poses = tracker.track_video_poses(poses)
        # print(f"트래킹된 프레임 수: {len(tracked_poses)}")
        
        # 정리
        tracker.cleanup()
    else:
        print("트래커 초기화 실패!")


def basic_scoring_example():
    """기본 점수 계산 예제"""
    print("\n=== 기본 점수 계산 예제 ===")
    
    # 설정 생성
    scoring_config = ScoringConfig(
        scorer_name='region_based',
        movement_weight=0.3,
        position_weight=0.2,
        interaction_weight=0.1,
        temporal_consistency_weight=0.2,
        persistence_weight=0.2,
        quality_threshold=0.3,
        min_track_length=10
    )
    
    # 점수 계산기 생성
    scorer = factory.create_scorer('region_based', scoring_config.__dict__)
    
    if scorer.initialize_scorer():
        print("점수 계산기 초기화 성공!")
        
        # 점수 계산 (예제용)
        # scores = scorer.calculate_scores(tracked_poses)
        # print(f"계산된 트랙 수: {len(scores)}")
        
        # 정리
        scorer.cleanup()
    else:
        print("점수 계산기 초기화 실패!")


def basic_classification_example():
    """기본 행동 분류 예제"""
    print("\n=== 기본 행동 분류 예제 ===")
    
    # 설정 생성
    classification_config = ActionClassificationConfig(
        model_name='stgcn',
        config_file='path/to/stgcn_config.py',
        model_path='path/to/stgcn_checkpoint.pth',
        window_size=100,
        class_names=['NonFight', 'Fight'],
        device='cuda:0'
    )
    
    # 분류기 생성
    classifier = factory.create_classifier('stgcn', classification_config.__dict__)
    
    if classifier.initialize_model():
        print("분류기 초기화 성공!")
        
        # 분류 실행 (예제용)
        # results = classifier.classify_multiple_windows(windows)
        # print(f"분류된 윈도우 수: {len(results)}")
        
        # 정리
        classifier.cleanup()
    else:
        print("분류기 초기화 실패!")


def unified_pipeline_example():
    """통합 파이프라인 예제"""
    print("\n=== 통합 파이프라인 예제 ===")
    
    # 각 단계별 설정
    pose_config = PoseEstimationConfig(
        model_name='rtmo',
        config_file='path/to/rtmo_config.py',
        model_path='path/to/rtmo_checkpoint.pth'
    )
    
    tracking_config = TrackingConfig(
        tracker_name='bytetrack',
        frame_rate=30
    )
    
    scoring_config = ScoringConfig(
        scorer_name='region_based'
    )
    
    classification_config = ActionClassificationConfig(
        model_name='stgcn',
        config_file='path/to/stgcn_config.py',
        model_path='path/to/stgcn_checkpoint.pth',
        window_size=100
    )
    
    # 파이프라인 설정
    pipeline_config = PipelineConfig(
        pose_config=pose_config,
        tracking_config=tracking_config,
        scoring_config=scoring_config,
        classification_config=classification_config,
        window_size=100,
        window_stride=50
    )
    
    try:
        # 파이프라인 생성
        with UnifiedPipeline(pipeline_config) as pipeline:
            print("통합 파이프라인 초기화 성공!")
            
            # 비디오 처리 (예제용 - 실제 파일 경로 필요)
            # result = pipeline.process_video('path/to/test_video.mp4')
            # print(f"처리 완료: {result.total_frames} 프레임, {result.processed_windows} 윈도우")
            # print(f"평균 FPS: {result.avg_fps:.1f}")
            
            # 성능 정보 출력
            pipeline_info = pipeline.get_pipeline_info()
            print("파이프라인 구성:")
            for module_type, info in pipeline_info['modules'].items():
                if info:
                    print(f"  {module_type}: {info.get('model_name', 'N/A')}")
    
    except Exception as e:
        print(f"파이프라인 에러: {str(e)}")


def annotation_pipeline_example():
    """어노테이션 파이프라인 예제"""
    print("\n=== 어노테이션 파이프라인 예제 ===")
    
    # 어노테이션 설정
    annotation_config = AnnotationConfig(
        pose_config=PoseEstimationConfig(
            model_name='rtmo',
            config_file='path/to/rtmo_config.py',
            model_path='path/to/rtmo_checkpoint.pth'
        ),
        tracking_config=TrackingConfig(
            tracker_name='bytetrack',
            frame_rate=30
        ),
        output_format='json',
        min_pose_confidence=0.3,
        min_track_length=10
    )
    
    try:
        # 어노테이션 파이프라인 생성
        annotation_pipeline = AnnotationPipeline(annotation_config)
        print("어노테이션 파이프라인 초기화 성공!")
        
        # 어노테이션 데이터 생성 (예제용)
        # annotation_data = annotation_pipeline.process_video_for_annotation(
        #     'path/to/video.mp4', 
        #     label='Fight'
        # )
        # 
        # # 결과 저장
        # annotation_pipeline.save_annotation_data(
        #     annotation_data, 
        #     'output/annotation_data'
        # )
        
        # 통계 출력
        stats = annotation_pipeline.get_pipeline_stats()
        print(f"처리된 비디오: {stats['total_videos']}")
        
        # 정리
        annotation_pipeline.cleanup()
    
    except Exception as e:
        print(f"어노테이션 파이프라인 에러: {str(e)}")


def visualization_example():
    """시각화 예제"""
    print("\n=== 시각화 예제 ===")
    
    # 포즈 시각화
    pose_visualizer = PoseVisualizer(
        show_bbox=True,
        show_keypoints=True,
        show_skeleton=True,
        show_track_id=True
    )
    print("포즈 시각화 도구 생성 완료")
    
    # 결과 시각화
    result_visualizer = ResultVisualizer(figsize=(12, 8))
    print("결과 시각화 도구 생성 완료")
    
    # 비디오 시각화 (예제용)
    # pose_visualizer.visualize_video_poses(
    #     'input_video.mp4',
    #     frame_poses_list,
    #     output_path='output_video.mp4'
    # )


def factory_info_example():
    """팩토리 정보 예제"""
    print("\n=== 팩토리 정보 예제 ===")
    
    # 등록된 모듈 확인
    all_modules = factory.list_all_modules()
    
    print("등록된 모듈들:")
    for category, modules in all_modules.items():
        print(f"\n{category}:")
        for module_name in modules:
            print(f"  - {module_name}")
    
    # 특정 모듈 정보 조회
    if factory.validate_module('pose_estimator', 'rtmo'):
        module_info = factory.get_module_info('pose_estimator', 'rtmo')
        print(f"\nRTMO 모듈 정보:")
        print(f"  클래스: {module_info['class'].__name__}")
        print(f"  모듈 경로: {module_info['module']}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("⚠️  이 예제 파일은 더 이상 실행하지 마세요!")
    print("="*60)
    print("\n대신 다음과 같이 통합 메인 실행기를 사용하세요:")
    print()
    print("# 기본 추론")
    print("python recognizer/main.py --mode inference --input video.mp4")
    print()
    print("# PKL 생성 + 시각화")
    print("python recognizer/main.py --mode inference --input video.mp4 --enable_evaluation --enable_visualization")
    print()
    print("# 분리 파이프라인")
    print("python recognizer/main.py --mode separated --input data/videos")
    print()
    print("# PKL 시각화")
    print("python recognizer/main.py --mode annotation --pkl_file stage2.pkl --video_file video.mp4")
    print()
    print("자세한 사용법은 USAGE.md를 참고하세요!")
    print("="*60)
    
    # 참고용으로 팩토리 정보만 표시
    print("\n참고: 사용 가능한 모듈들")
    factory_info_example()