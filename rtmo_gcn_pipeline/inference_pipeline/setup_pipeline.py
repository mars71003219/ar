#!/usr/bin/env python3
"""
Pipeline Setup Script
파이프라인 설정 및 초기화 스크립트
"""

import os
import os.path as osp
import sys
import logging
from pathlib import Path
import json

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_annotations():
    """샘플 어노테이션 파일 생성"""
    annotations_content = """# Sample annotations file for RWF-2000 test data
# Format: video_name,label (0: NonFight, 1: Fight)
Fight_1.mp4,1
Fight_2.mp4,1
Fight_3.mp4,1
NonFight_1.mp4,0
NonFight_2.mp4,0
NonFight_3.mp4,0
"""
    
    sample_annotations_path = "./sample_annotations.txt"
    with open(sample_annotations_path, 'w', encoding='utf-8') as f:
        f.write(annotations_content)
    
    logger.info(f"샘플 어노테이션 파일 생성: {sample_annotations_path}")
    return sample_annotations_path

def create_sample_label_map():
    """샘플 라벨 매핑 파일 생성"""
    label_map_content = """Fight: 1
NonFight: 0
"""
    
    sample_label_map_path = "./sample_label_map.txt"
    with open(sample_label_map_path, 'w', encoding='utf-8') as f:
        f.write(label_map_content)
    
    logger.info(f"샘플 라벨 매핑 파일 생성: {sample_label_map_path}")
    return sample_label_map_path

def check_dependencies():
    """필수 의존성 확인"""
    required_packages = [
        'torch',
        'torchvision', 
        'cv2',
        'numpy',
        'mmpose',
        'mmaction2',
        'mmengine',
        'mmcv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            else:
                __import__(package)
            logger.info(f"✓ {package} 설치됨")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"✗ {package} 누락")
    
    if missing_packages:
        logger.error(f"누락된 패키지: {missing_packages}")
        return False
    
    logger.info("모든 필수 패키지가 설치되어 있습니다")
    return True

def setup_directories():
    """필요한 디렉토리 생성"""
    directories = [
        "./results",
        "./results/overlays",
        "./results/individual_results",
        "./logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"디렉토리 생성: {directory}")

def validate_model_paths():
    """모델 파일 경로 검증"""
    from config import POSE_CONFIG, POSE_CHECKPOINT, GCN_CONFIG, GCN_CHECKPOINT
    
    model_files = [
        ("RTMO Config", POSE_CONFIG),
        ("RTMO Checkpoint", POSE_CHECKPOINT),
        ("STGCN++ Config", GCN_CONFIG),
        ("STGCN++ Checkpoint", GCN_CHECKPOINT)
    ]
    
    missing_files = []
    
    for name, path in model_files:
        if osp.exists(path):
            logger.info(f"✓ {name}: {path}")
        else:
            missing_files.append((name, path))
            logger.error(f"✗ {name}: {path} (파일 없음)")
    
    if missing_files:
        logger.error("누락된 모델 파일:")
        for name, path in missing_files:
            logger.error(f"  - {name}: {path}")
        return False
    
    logger.info("모든 모델 파일이 존재합니다")
    return True

def create_quick_test_script():
    """빠른 테스트 스크립트 생성"""
    test_script_content = """#!/usr/bin/env python3
\"\"\"
Quick Test Script
빠른 파이프라인 테스트 스크립트
\"\"\"

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import validate_config, check_gpu_availability

def main():
    print("=== 파이프라인 빠른 테스트 ===")
    
    # 1. 설정 검증
    print("1. 설정 검증...")
    if validate_config():
        print("✓ 설정 검증 통과")
    else:
        print("✗ 설정 검증 실패")
        return False
    
    # 2. GPU 확인
    print("2. GPU 사용 가능성 확인...")
    gpu_available = check_gpu_availability()
    if gpu_available:
        print("✓ GPU 사용 가능")
    else:
        print("! CPU 모드로 실행됩니다")
    
    # 3. 모듈 import 테스트
    print("3. 핵심 모듈 import 테스트...")
    try:
        from pose_estimator import RTMOPoseEstimator
        from fight_tracker import FightPrioritizedTracker
        from action_classifier import STGCNActionClassifier
        from metrics_calculator import MetricsCalculator
        from video_overlay import VideoOverlayGenerator
        from main_pipeline import EndToEndPipeline
        print("✓ 모든 모듈 import 성공")
    except Exception as e:
        print(f"✗ 모듈 import 실패: {e}")
        return False
    
    print("=== 빠른 테스트 완료 ===")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
"""
    
    test_script_path = "./quick_test.py"
    with open(test_script_path, 'w', encoding='utf-8') as f:
        f.write(test_script_content)
    
    os.chmod(test_script_path, 0o755)
    logger.info(f"빠른 테스트 스크립트 생성: {test_script_path}")
    return test_script_path

def create_config_template():
    """설정 템플릿 파일 생성"""
    config_template = {
        "description": "STGCN++ Violence Detection Pipeline Configuration Template",
        "paths": {
            "pose_config": "Path to RTMO model config file",
            "pose_checkpoint": "Path to RTMO model checkpoint",
            "gcn_config": "Path to STGCN++ model config file", 
            "gcn_checkpoint": "Path to STGCN++ model checkpoint",
            "input_dir": "Input video directory",
            "output_dir": "Output results directory"
        },
        "inference_settings": {
            "device": "cuda:0 or cpu",
            "batch_size": 8,
            "sequence_length": 30,
            "pose_score_threshold": 0.3,
            "confidence_threshold": 0.5
        },
        "fight_tracking": {
            "region_weights": {
                "center": 1.0,
                "top_left": 0.7,
                "top_right": 0.7,
                "bottom_left": 0.6,
                "bottom_right": 0.6
            },
            "composite_weights": {
                "position": 0.3,
                "movement": 0.25,
                "interaction": 0.25,
                "detection": 0.1,
                "consistency": 0.1
            }
        },
        "video_processing": {
            "max_frames": 900,
            "resize": [640, 480],
            "fps": "auto"
        },
        "overlay_settings": {
            "enabled": True,
            "joint_color": [0, 255, 0],
            "skeleton_color": [255, 0, 0],
            "text_color": [255, 255, 255],
            "font_scale": 1.0,
            "thickness": 2
        }
    }
    
    config_template_path = "./config_template.json"
    with open(config_template_path, 'w', encoding='utf-8') as f:
        json.dump(config_template, f, indent=2, ensure_ascii=False)
    
    logger.info(f"설정 템플릿 생성: {config_template_path}")
    return config_template_path

def main():
    """메인 설정 함수"""
    print("=== STGCN++ Violence Detection Pipeline 설정 ===")
    
    logger.info("파이프라인 설정 시작...")
    
    # 1. 의존성 확인
    logger.info("1. 의존성 확인 중...")
    if not check_dependencies():
        logger.error("의존성 확인 실패")
        return False
    
    # 2. 디렉토리 설정
    logger.info("2. 디렉토리 설정 중...")
    setup_directories()
    
    # 3. 모델 파일 검증
    logger.info("3. 모델 파일 검증 중...")
    model_valid = validate_model_paths()
    if not model_valid:
        logger.warning("일부 모델 파일이 누락되었습니다. 실행 전에 확인해주세요.")
    
    # 4. 샘플 파일 생성
    logger.info("4. 샘플 파일 생성 중...")
    create_sample_annotations()
    create_sample_label_map()
    create_config_template()
    
    # 5. 테스트 스크립트 생성
    logger.info("5. 테스트 스크립트 생성 중...")
    create_quick_test_script()
    
    print("\n=== 설정 완료 ===")
    print("다음 단계:")
    print("1. python quick_test.py - 빠른 테스트 실행")
    print("2. ./run_example.sh - 예제 실행")
    print("3. python run_inference.py --help - 사용법 확인")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)