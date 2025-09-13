#!/usr/bin/env python3
"""
쓰러짐 모듈 테스트 스크립트
"""

from main import register_modules
from utils.factory import ModuleFactory
from utils.config_loader import load_config
import os

def main():
    print('=== Testing Falldown Module Initialization ===')

    # 모듈 등록
    success = register_modules()
    print(f'Module registration: {"SUCCESS" if success else "FAILED"}')

    # 설정 로드
    config = load_config('test_basic_config.yaml', None, {})
    print(f'Config loaded: {config is not None}')

    if config:
        # FalldownScorer 테스트
        print('\n--- Testing FalldownScorer ---')
        try:
            scorer_config = config.get('pipeline', {}).get('scorer', {})
            scorer = ModuleFactory.create_scorer(scorer_config.get('name'), scorer_config)
            print(f'FalldownScorer created: {scorer is not None}')
            print(f'Scorer type: {type(scorer).__name__}')
            if scorer:
                scorer_info = scorer.get_scorer_info()
                print(f'Scorer info: {scorer_info.get("scorer_type")} - falldown_optimized: {scorer_info.get("falldown_optimized")}')
        except Exception as e:
            print(f'FalldownScorer creation failed: {e}')
        
        # 통합 STGCNClassifier 테스트 (falldown 모델로 설정됨)
        print('\n--- Testing STGCNClassifier (Falldown Model) ---')
        try:
            classifier_config = config.get('pipeline', {}).get('classifier', {})
            print(f'Classifier config: {classifier_config.get("name")}')
            print(f'Model path: {classifier_config.get("model_path")}')
            print(f'Config file: {classifier_config.get("config_file")}')
            
            # 모델 파일 존재 확인만 (실제 초기화는 GPU 메모리가 많이 필요)
            model_exists = os.path.exists(classifier_config.get('model_path', ''))
            config_exists = os.path.exists(classifier_config.get('config_file', ''))
            print(f'Model file exists: {model_exists}')
            print(f'Config file exists: {config_exists}')
            
            if model_exists and config_exists:
                print('STGCNClassifier setup: READY (files found)')
            else:
                print('STGCNClassifier setup: FILES MISSING')
                
        except Exception as e:
            print(f'STGCNClassifier test failed: {e}')

    print('\n=== Test Complete ===')

if __name__ == "__main__":
    main()