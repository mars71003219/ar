# Recognizer 프로젝트 문서 모음

## 개요
Recognizer는 MMPose 기반의 실시간 동작 인식 및 분석 시스템입니다. 이 문서 모음은 시스템의 구조, 설계, 사용법을 상세히 설명합니다.

## 문서 목록

### 1. [폴더 구조별 기능 설명서](01_folder_structure_guide.md)
- **목적**: 프로젝트의 전체 폴더 구조와 각 디렉토리의 역할 설명
- **내용**:
  - 15개 주요 디렉토리별 상세 기능 설명
  - 모듈별 책임과 역할
  - 데이터 플로우 개요
  - 주요 특징 및 확장성

### 2. [Stage별 PKL 데이터 구조 설명서](02_pkl_data_structure_guide.md)
- **목적**: 3단계 파이프라인에서 생성되는 PKL 데이터의 구조 상세 설명
- **내용**:
  - Stage1: 포즈 추정 결과 (FramePoses, Person 클래스)
  - Stage2: 추적 결과 (track_id 추가, 스코어링 정보)
  - Stage3: 최종 데이터셋 (STGCN 호환 텐서 형식)
  - 데이터 변환 과정 및 접근 방법
  - 실제 사용 예시 코드

### 3. [모드별 파이프라인 아키텍처 설명서](03_pipeline_architecture_guide.md)
- **목적**: 시스템의 다양한 실행 모드와 파이프라인 구조 설명
- **내용**:
  - Annotation 모드 (Stage1-3)
  - Inference 모드 (Analysis, Realtime, Visualize)
  - 파이프라인 구현체 (DualService, Separated, Analysis)
  - 모듈 간 데이터 플로우
  - 성능 최적화 전략

### 4. [UML 다이어그램 설계 문서](04_uml_diagrams.md)
- **목적**: 시스템의 구조와 동작을 시각적으로 표현
- **내용**:
  - 클래스 다이어그램: 전체 시스템 클래스 구조 및 관계
  - 시퀀스 다이어그램: 실시간 처리, Annotation 파이프라인, 이벤트 감지, 멀티프로세스 처리
  - 상태 다이어그램: 이벤트 상태 전이, 파이프라인 실행 상태
  - Mermaid 형식으로 작성되어 GitHub에서 바로 렌더링 가능

### 5. [Config 설정 정의 및 모드별 셋팅 가이드](05_config_settings_guide.md)
- **목적**: config.yaml의 모든 설정 항목 설명 및 모드별 최적 설정 가이드
- **내용**:
  - 전체 설정 구조 및 계층
  - 모드별 상세 설정 (Annotation, Inference)
  - 모델 설정 (포즈 추정, 동작 분류, 추적)
  - 성능 최적화 설정
  - 이벤트 관리 및 파일 관리 설정
  - 환경별 권장 설정 (개발/프로덕션/대용량 처리)

### 6. [Pose 및 STGCN 학습 설정 설명 및 가이드](06_pose_stgcn_training_guide.md)
- **목적**: 포즈 추정 모델과 동작 분류 모델의 학습 설정 및 최적화 방법
- **내용**:
  - RTMO 포즈 추정 모델 설정 (PyTorch, ONNX, TensorRT)
  - STGCN++ 동작 분류 모델 아키텍처 및 설정
  - Fight/Falldown 감지 모델별 특화 설정
  - 학습 하이퍼파라미터 튜닝 가이드
  - 모델 변환 과정 (PyTorch → ONNX → TensorRT)
  - 성능 최적화 및 디버깅 방법

### 7. [최종 소프트웨어 설계서](07_software_design_document.md)
- **목적**: 전체 시스템의 종합적인 설계 문서
- **내용**:
  - 시스템 개요 및 목적
  - 전체 시스템 아키텍처 (4계층 구조)
  - 모듈별 상세 설계 (포즈 추정, 추적, 분류, 이벤트, 시각화)
  - 파이프라인 설계 및 데이터 설계
  - 성능 설계 및 최적화 전략
  - 품질 보증 및 테스트 전략
  - 보안 및 안정성 설계
  - 배포 및 운영 설계
  - 확장성 및 유지보수 설계

## 문서 간 연관성

```
07_software_design_document.md (전체 설계 개요)
        │
        ├── 01_folder_structure_guide.md (구현 구조)
        │
        ├── 03_pipeline_architecture_guide.md (파이프라인 설계)
        │   └── 04_uml_diagrams.md (시각적 설계)
        │
        ├── 02_pkl_data_structure_guide.md (데이터 설계)
        │
        ├── 05_config_settings_guide.md (설정 가이드)
        │
        └── 06_pose_stgcn_training_guide.md (모델 학습)
```

## 읽기 권장 순서

### 처음 사용하는 경우
1. **07_software_design_document.md** - 전체 시스템 이해
2. **01_folder_structure_guide.md** - 코드 구조 파악
3. **05_config_settings_guide.md** - 기본 설정 방법
4. **03_pipeline_architecture_guide.md** - 실행 모드 이해

### 개발자인 경우
1. **04_uml_diagrams.md** - 클래스 구조 이해
2. **02_pkl_data_structure_guide.md** - 데이터 형식 이해
3. **01_folder_structure_guide.md** - 모듈별 구현 위치
4. **06_pose_stgcn_training_guide.md** - 모델 학습 방법

### 운영자인 경우
1. **05_config_settings_guide.md** - 설정 최적화
2. **03_pipeline_architecture_guide.md** - 파이프라인 운영
3. **07_software_design_document.md** (섹션 9) - 배포 및 운영

### 연구자인 경우
1. **06_pose_stgcn_training_guide.md** - 모델 학습 및 튜닝
2. **02_pkl_data_structure_guide.md** - 데이터 형식 이해
3. **04_uml_diagrams.md** - 시스템 구조 분석

## 빠른 시작 가이드

### 1. 실시간 처리 실행
```bash
# config.yaml에서 mode 설정
mode: inference.realtime

# Docker 컨테이너에서 실행
docker exec mmlabs bash -c "cd /workspace/recognizer && python3 main.py"
```

### 2. 데이터 준비 (3단계)
```bash
# Stage1: 포즈 추정
docker exec mmlabs bash -c "cd /workspace/recognizer && python3 main.py --mode annotation.stage1"

# Stage2: 추적
docker exec mmlabs bash -c "cd /workspace/recognizer && python3 main.py --mode annotation.stage2"

# Stage3: 데이터셋 생성
docker exec mmlabs bash -c "cd /workspace/recognizer && python3 main.py --mode annotation.stage3"
```

### 3. 모델 학습
```bash
# STGCN++ Fight 모델 학습
cd /workspace/mmaction2
python tools/train.py configs/skeleton/stgcnpp/stgcnpp_enhanced_fight_detection_stable.py
```

## 주요 설정 파일

### 메인 설정
- **`recognizer/configs/config.yaml`** - 전체 시스템 설정
- **`recognizer/main.py`** - 메인 실행 파일

### 모델 설정
- **`mmaction2/configs/skeleton/stgcnpp/stgcnpp_enhanced_fight_detection_stable.py`** - Fight 감지 모델
- **`mmaction2/configs/skeleton/stgcnpp/stgcnpp_enhanced_falldown_detection_stable.py`** - Falldown 감지 모델

## 시스템 요구사항

### 하드웨어
- **GPU**: NVIDIA RTX 3090 이상 (24GB VRAM 권장)
- **CPU**: 8코어 이상
- **RAM**: 32GB 이상
- **Storage**: 1TB 이상 (데이터셋 저장용)

### 소프트웨어
- **OS**: Ubuntu 20.04 LTS
- **Docker**: NVIDIA Docker 지원
- **CUDA**: 11.8 이상
- **Python**: 3.8 이상

## 문제 해결

### 일반적인 문제
1. **GPU 메모리 부족**: `05_config_settings_guide.md`의 메모리 최적화 섹션 참조
2. **학습이 안 됨**: `06_pose_stgcn_training_guide.md`의 문제 해결 섹션 참조
3. **설정 오류**: `05_config_settings_guide.md`의 설정 검증 섹션 참조

### 성능 최적화
- **실시간 처리**: `03_pipeline_architecture_guide.md`의 성능 최적화 섹션
- **대용량 처리**: `05_config_settings_guide.md`의 대용량 데이터 처리 설정
- **모델 최적화**: `06_pose_stgcn_training_guide.md`의 성능 최적화 섹션

## 지원 및 문의

- **기술 문서**: 위 7개 문서 참조
- **코드 구조**: `01_folder_structure_guide.md` 참조
- **설정 문제**: `05_config_settings_guide.md` 참조
- **모델 학습**: `06_pose_stgcn_training_guide.md` 참조

---
