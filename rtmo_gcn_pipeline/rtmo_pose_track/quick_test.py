#!/usr/bin/env python3
"""
Quick Test Script for Enhanced STGCN++ Annotation Generator
개선된 STGCN++ 어노테이션 생성기 빠른 테스트 스크립트

이 스크립트는 실제 실행 전 환경 및 기능을 빠르게 테스트합니다.
"""

import os
import sys
import numpy as np
import pickle
import tempfile
from datetime import datetime

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_dependencies():
    """의존성 체크"""
    print("1. 의존성 체크 중...")
    
    dependencies = {
        'torch': 'PyTorch',
        'cv2': 'OpenCV',
        'mmcv': 'MMCV',
        'mmengine': 'MMEngine', 
        'scipy': 'SciPy',
        'tqdm': 'TQDM',
        'numpy': 'NumPy'
    }
    
    missing = []
    versions = {}
    
    for module, name in dependencies.items():
        try:
            imported = __import__(module)
            version = getattr(imported, '__version__', 'unknown')
            versions[name] = version
            print(f"  {name}: {version}")
        except ImportError:
            missing.append(name)
            print(f"  {name}: 누락됨")
    
    if missing:
        print(f"\n 누락된 의존성: {', '.join(missing)}")
        print("다음 명령어로 설치하세요:")
        print("pip install torch opencv-python mmcv mmengine scipy tqdm numpy")
        return False
    
    print("  Pass: 모든 의존성 확인됨!")
    return True

def test_gpu_availability():
    """GPU 사용 가능성 테스트"""
    print("\n2. GPU 사용 가능성 체크 중...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / 1e9
            
            print(f"  GPU 사용 가능")
            print(f"  GPU 개수: {gpu_count}")
            print(f"  현재 GPU: {gpu_name}")
            print(f"  GPU 메모리: {gpu_memory:.1f}GB")
            
            # GPU 메모리 테스트
            try:
                test_tensor = torch.randn(1000, 1000).cuda()
                del test_tensor
                torch.cuda.empty_cache()
                print(f"  Pass: GPU 메모리 할당 테스트 통과")
            except Exception as e:
                print(f"  Fail:  GPU 메모리 할당 실패: {e}")
                
            return True
        else:
            print("  GPU 사용 불가능 (CPU 모드로 실행)")
            return False
            
    except ImportError:
        print("  PyTorch 누락됨")
        return False

def test_core_classes():
    """핵심 클래스들 테스트"""
    print("\n3. 핵심 클래스 기능 테스트 중...")
    
    try:
        # 5영역 분할 테스트
        from enhanced_rtmo_bytetrack_pose_extraction import RegionBasedPositionScorer
        
        scorer = RegionBasedPositionScorer(640, 480)
        print(f"  RegionBasedPositionScorer 초기화 성공")
        print(f"  영역 개수: {len(scorer.regions)}")
        
        # 가짜 bbox 히스토리로 테스트
        fake_bbox_history = [
            [100, 100, 200, 200],  # 좌상단
            [500, 300, 600, 400],  # 우하단
            [300, 200, 400, 300]   # 중앙
        ]
        
        position_score, region_breakdown = scorer.calculate_position_score(fake_bbox_history)
        print(f"  위치 점수 계산 성공: {position_score:.3f}")
        print(f"  영역별 분석: {len(region_breakdown)}개 영역")
        
        # 적응적 가중치 학습 테스트
        from enhanced_rtmo_bytetrack_pose_extraction import AdaptiveRegionImportance
        
        adaptive = AdaptiveRegionImportance()
        print(f"  AdaptiveRegionImportance 초기화 성공")
        
        # 복합 점수 계산기 테스트
        from enhanced_rtmo_bytetrack_pose_extraction import EnhancedFightInvolvementScorer
        
        fight_scorer = EnhancedFightInvolvementScorer((480, 640), enable_adaptive=False)
        print(f"  EnhancedFightInvolvementScorer 초기화 성공")
        
        return True
        
    except Exception as e:
        print(f"  핵심 클래스 테스트 실패: {e}")
        return False

def test_pickle_operations():
    """PKL 파일 읽기/쓰기 테스트"""
    print("\n4. PKL 파일 작업 테스트 중...")
    
    try:
        # 테스트 데이터 생성
        test_annotation = {
            'total_persons': 2,
            'video_info': {
                'frame_dir': 'test_video',
                'total_frames': 60,
                'img_shape': [480, 640],
                'label': 1
            },
            'persons': {
                'person_00': {
                    'track_id': 1,
                    'composite_score': 0.85,
                    'score_breakdown': {
                        'movement': 0.9,
                        'position': 0.8,
                        'interaction': 0.7,
                        'temporal_consistency': 0.95,
                        'persistence': 0.9
                    },
                    'region_breakdown': {
                        'top_left': 0.1,
                        'top_right': 0.2,
                        'bottom_left': 0.3,
                        'bottom_right': 0.2,
                        'center_overlap': 0.9
                    },
                    'track_quality': 0.78,
                    'rank': 1,
                    'annotation': {
                        'keypoint': np.random.randn(1, 60, 17, 2).astype(np.float32),
                        'keypoint_score': np.random.rand(1, 60, 17).astype(np.float32),
                        'num_keypoints': 17,
                        'track_id': 1
                    }
                }
            },
            'score_weights': {
                'movement_intensity': 0.30,
                'position_5region': 0.35,
                'interaction': 0.20,
                'temporal_consistency': 0.10,
                'persistence': 0.05
            }
        }
        
        # 임시 파일에 저장
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            
        with open(tmp_path, 'wb') as f:
            pickle.dump(test_annotation, f)
        
        print(f"  PKL 파일 저장 성공")
        
        # 파일 읽기
        with open(tmp_path, 'rb') as f:
            loaded_annotation = pickle.load(f)
        
        print(f"  PKL 파일 로드 성공")
        
        # 데이터 무결성 체크
        assert loaded_annotation['total_persons'] == test_annotation['total_persons']
        assert 'person_00' in loaded_annotation['persons']
        
        keypoints = loaded_annotation['persons']['person_00']['annotation']['keypoint']
        assert keypoints.shape == (1, 60, 17, 2)
        
        print(f"  데이터 무결성 체크 통과")
        print(f"  키포인트 모양: {keypoints.shape}")
        
        # 임시 파일 삭제
        os.unlink(tmp_path)
        
        return True
        
    except Exception as e:
        print(f"  PKL 작업 테스트 실패: {e}")
        return False

# Global worker function for multiprocessing (must be at module level for pickling)
def _test_worker(x):
    """테스트용 워커 함수 - 제곱 계산"""
    return x * x

def test_parallel_processing():
    """병렬 처리 기능 테스트"""
    print("\n5. 병렬 처리 기능 테스트 중...")
    
    try:
        import multiprocessing as mp
        
        cpu_count = mp.cpu_count()
        print(f"  CPU 코어 수: {cpu_count}")
        
        # 간단한 병렬 테스트
        from concurrent.futures import ProcessPoolExecutor
        
        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(_test_worker, i) for i in range(4)]
            results = [f.result() for f in futures]
        
        expected = [0, 1, 4, 9]
        assert results == expected
        
        print(f"  병렬 처리 테스트 통과")
        
        # 리소스 모니터링 테스트
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            print(f"  리소스 모니터링 가능")
            print(f"  현재 CPU 사용률: {cpu_percent:.1f}%")
            print(f"  현재 메모리 사용률: {memory.percent:.1f}%")
            
        except ImportError:
            print(f"  psutil 누락 (리소스 모니터링 비활성화)")
        
        return True
        
    except Exception as e:
        print(f"  병렬 처리 테스트 실패: {e}")
        return False

def test_file_operations():
    """파일 작업 테스트"""
    print("\n6. 파일 작업 테스트 중...")
    
    try:
        # 임시 디렉토리 생성
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"  임시 디렉토리 생성: {temp_dir}")
            
            # 하위 디렉토리 구조 생성
            train_dir = os.path.join(temp_dir, 'train', 'Fight')
            os.makedirs(train_dir, exist_ok=True)
            
            # 가짜 비디오 파일 생성
            fake_video = os.path.join(train_dir, 'test_video.mp4')
            with open(fake_video, 'w') as f:
                f.write("fake video content")
            
            print(f"  가짜 비디오 파일 생성")
            
            # 파일 찾기 테스트
            from enhanced_rtmo_bytetrack_pose_extraction import find_video_files
            
            video_files = find_video_files(temp_dir)
            assert len(video_files) == 1
            assert video_files[0] == fake_video
            
            print(f"  비디오 파일 찾기 테스트 통과")
            
            # 출력 경로 생성 테스트
            from enhanced_rtmo_bytetrack_pose_extraction import get_output_path
            
            output_path = get_output_path(fake_video, temp_dir, temp_dir, '_test.pkl')
            
            print(f"  출력 경로 생성 테스트 통과")
            print(f"  예상 출력: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"  파일 작업 테스트 실패: {e}")
        return False

def run_comprehensive_test():
    """종합 테스트 실행"""
    print("Enhanced STGCN++ Annotation Generator - 종합 테스트")
    print("=" * 60)
    print(f"테스트 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tests = [
        ("의존성 체크", check_dependencies),
        ("GPU 사용 가능성", test_gpu_availability),
        ("핵심 클래스 기능", test_core_classes),
        ("PKL 파일 작업", test_pickle_operations),
        ("병렬 처리 기능", test_parallel_processing),
        ("파일 작업", test_file_operations)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  {test_name} 테스트 중 예외 발생: {e}")
            failed += 1
        
        print()
    
    print("=" * 60)
    print(" 테스트 결과 요약")
    print("=" * 60)
    print(f"통과: {passed}")
    print(f"실패: {failed}")
    print(f"성공률: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n 모든 테스트 통과! 시스템이 정상적으로 작동할 준비가 되었습니다.")
        print("\n다음 명령어로 실제 처리를 시작할 수 있습니다:")
        print("python run_enhanced_annotation.py demo config.py checkpoint.pth --input /path/to/videos")
        return True
    else:
        print(f"\n {failed}개 테스트 실패. 문제를 해결한 후 다시 시도하세요.")
        return False

def main():
    """메인 함수"""
    try:
        success = run_comprehensive_test()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n  사용자에 의해 테스트가 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n 테스트 중 예상치 못한 오류가 발생했습니다: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()