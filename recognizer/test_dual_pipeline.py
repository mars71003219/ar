#!/usr/bin/env python3
"""
듀얼 서비스 파이프라인 테스트 스크립트
"""

from main import register_modules
from utils.config_loader import load_config
from pipelines.dual_service import create_dual_service_pipeline

def main():
    print('=== Testing Dual Service Pipeline ===')

    # 모듈 등록
    success = register_modules()
    print(f'Module registration: {"SUCCESS" if success else "FAILED"}')

    # 설정 로드
    config = load_config('configs/config.yaml', None, {})
    print(f'Config loaded: {config is not None}')

    if config:
        # 듀얼 서비스 설정 확인
        dual_config = config.get('dual_service', {})
        print(f'Dual service enabled: {dual_config.get("enabled", False)}')
        print(f'Services: {dual_config.get("services", [])}')
        
        try:
            # 듀얼 서비스 파이프라인 생성 테스트
            print('\n--- Testing Dual Service Pipeline Creation ---')
            pipeline = create_dual_service_pipeline(config)
            
            if pipeline:
                print('Dual service pipeline created: SUCCESS')
                pipeline_info = pipeline.get_pipeline_info()
                print(f'Pipeline type: {pipeline_info.get("pipeline_type")}')
                print(f'Services: {pipeline_info.get("services")}')
                print(f'Modules initialized: {pipeline_info.get("modules_initialized")}')
                
                # 정리
                pipeline.cleanup()
                print('Pipeline cleanup: SUCCESS')
            else:
                print('Dual service pipeline created: FAILED')
                
        except Exception as e:
            print(f'Dual service pipeline test failed: {e}')
            import traceback
            traceback.print_exc()

    print('\n=== Test Complete ===')

if __name__ == "__main__":
    main()