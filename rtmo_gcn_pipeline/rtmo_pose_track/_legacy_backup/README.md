# Legacy Backup Files

이 폴더에는 구조화 이전의 원본 파일들이 백업되어 있습니다.

## 백업된 파일들

- `enhanced_rtmo_bytetrack_pose_extraction.py` - 메인 포즈 추출 및 트래킹 로직 (4,430줄)
- `error_logger.py` - 에러 로깅 시스템
- `inference_pipeline.py` - 추론 파이프라인 원본
- `separated_pose_pipeline.py` - 분리된 포즈 파이프라인 원본  
- `run_pose_extraction.py` - 통합 실행 스크립트 원본
- `visualizer.py` - 시각화 시스템 원본
- `unified_pose_processor.py` - 통합 처리기 원본
- `test_data copy/` - 중복된 테스트 데이터

## 새로운 구조에서의 위치

- `enhanced_rtmo_bytetrack_pose_extraction.py` → `core/` 모듈로 분할됨
- `error_logger.py` → `logging/error_logger.py`
- `inference_pipeline.py` → `scripts/inference_pipeline.py`
- `separated_pose_pipeline.py` → `scripts/separated_pose_pipeline.py`
- `run_pose_extraction.py` → `scripts/run_pose_extraction.py`
- `visualizer.py` → `visualization/visualizer.py`

## 주의사항

- 이 파일들은 호환성을 위해 백업되었습니다
- 새로운 구조의 scripts 폴더에서 필요시 이 파일들을 참조합니다
- 삭제하지 마세요 - 기존 기능 보존을 위해 필요합니다