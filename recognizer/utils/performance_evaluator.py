"""
성능 평가 및 결과 저장 유틸리티

추론 파이프라인에서 선택적으로 사용할 수 있는 성능 지표 계산 및 결과 저장 기능을 제공합니다.
"""

import json
import pickle
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

from .file_utils import ensure_directory
from .data_structure import ClassificationResult, FramePoses


@dataclass
class PerformanceMetrics:
    """성능 지표 데이터"""
    # 기본 분류 성능
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    
    # 혼동 행렬
    confusion_matrix: np.ndarray
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    
    # 클래스별 성능
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    per_class_f1: Dict[str, float]
    
    # 추가 통계
    total_samples: int
    class_distribution: Dict[str, int]
    confidence_stats: Dict[str, float]
    
    # 상세 리포트
    classification_report: str
    
    # 처리 성능
    processing_stats: Dict[str, float] = field(default_factory=dict)


@dataclass
class InferenceResult:
    """단일 추론 결과"""
    sample_id: str
    true_label: Optional[str]
    predicted_label: str
    confidence: float
    class_probabilities: Dict[str, float]
    processing_time: float
    
    # 메타데이터
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OverlayData:
    """오버레이 시각화용 데이터"""
    sample_id: str
    video_path: Optional[str]
    
    # 포즈 및 트래킹 데이터
    pose_data: List[FramePoses]
    tracking_data: Optional[List[FramePoses]]
    score_data: Optional[Dict[str, Any]]
    
    # 분류 결과
    window_results: List[ClassificationResult]
    final_prediction: str
    final_confidence: float
    
    # 시각화 설정
    frame_annotations: List[Dict[str, Any]] = field(default_factory=list)


class PerformanceEvaluator:
    """성능 평가 및 결과 저장 클래스"""
    
    def __init__(self, output_dir: str = "output/evaluation", 
                 class_names: List[str] = None):
        """
        Args:
            output_dir: 결과 저장 디렉토리
            class_names: 클래스 이름 리스트
        """
        self.output_dir = Path(output_dir)
        self.class_names = class_names or ["NonFight", "Fight"]
        
        # 결과 저장용 리스트
        self.inference_results: List[InferenceResult] = []
        self.overlay_data_list: List[OverlayData] = []
        
        # 출력 디렉토리 초기화
        self._initialize_directories()
        
        logging.info(f"PerformanceEvaluator initialized with output_dir: {output_dir}")
    
    def _initialize_directories(self):
        """출력 디렉토리 초기화"""
        ensure_directory(str(self.output_dir))
        ensure_directory(str(self.output_dir / "results"))
        ensure_directory(str(self.output_dir / "overlay_data"))
        ensure_directory(str(self.output_dir / "performance_metrics"))
        ensure_directory(str(self.output_dir / "confusion_matrices"))
    
    def add_inference_result(self, result: InferenceResult):
        """추론 결과 추가"""
        self.inference_results.append(result)
    
    def add_overlay_data(self, overlay: OverlayData):
        """오버레이 데이터 추가"""
        self.overlay_data_list.append(overlay)
    
    def calculate_performance_metrics(self) -> PerformanceMetrics:
        """성능 지표 계산"""
        if not self.inference_results:
            logging.warning("No inference results available for metric calculation")
            return self._create_empty_metrics()
        
        # 라벨이 있는 결과만 필터링 (성능 평가용)
        labeled_results = [r for r in self.inference_results if r.true_label is not None]
        
        if not labeled_results:
            logging.warning("No labeled results available for performance evaluation")
            return self._create_empty_metrics()
        
        # 라벨 추출
        y_true = [r.true_label for r in labeled_results]
        y_pred = [r.predicted_label for r in labeled_results]
        
        # 기본 성능 지표
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted', labels=self.class_names, zero_division=0)
        
        # 혼동 행렬
        cm = confusion_matrix(y_true, y_pred, labels=self.class_names)
        
        # 분류 리포트
        report = classification_report(
            y_true, y_pred, 
            labels=self.class_names, 
            output_dict=True, 
            zero_division=0
        )
        
        # 클래스별 성능
        per_class_precision = {}
        per_class_recall = {}
        per_class_f1 = {}
        
        for class_name in self.class_names:
            if class_name in report:
                per_class_precision[class_name] = report[class_name]['precision']
                per_class_recall[class_name] = report[class_name]['recall']
                per_class_f1[class_name] = report[class_name]['f1-score']
            else:
                per_class_precision[class_name] = 0.0
                per_class_recall[class_name] = 0.0
                per_class_f1[class_name] = 0.0
        
        # 혼동 행렬 세부 값 (2클래스 가정)
        if len(self.class_names) == 2:
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            overall_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            overall_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        else:
            # 다중 클래스의 경우 가중 평균 사용
            tp = sum(cm[i, i] for i in range(len(self.class_names)))
            fp = sum(cm[:, i].sum() - cm[i, i] for i in range(len(self.class_names)))
            fn = sum(cm[i, :].sum() - cm[i, i] for i in range(len(self.class_names)))
            tn = cm.sum() - tp - fp - fn
            overall_precision = report['weighted avg']['precision']
            overall_recall = report['weighted avg']['recall']
        
        # 클래스 분포
        unique, counts = np.unique(y_true, return_counts=True)
        class_distribution = dict(zip(unique, counts))
        
        # 신뢰도 통계
        confidences = [r.confidence for r in labeled_results if r.confidence > 0]
        confidence_stats = {
            'mean': float(np.mean(confidences)) if confidences else 0.0,
            'std': float(np.std(confidences)) if confidences else 0.0,
            'min': float(np.min(confidences)) if confidences else 0.0,
            'max': float(np.max(confidences)) if confidences else 0.0,
            'median': float(np.median(confidences)) if confidences else 0.0
        }
        
        # 처리 성능 통계
        processing_times = [r.processing_time for r in self.inference_results]
        processing_stats = {
            'total_samples': len(self.inference_results),
            'avg_processing_time': float(np.mean(processing_times)) if processing_times else 0.0,
            'total_processing_time': float(np.sum(processing_times)) if processing_times else 0.0,
            'fps': len(processing_times) / np.sum(processing_times) if np.sum(processing_times) > 0 else 0.0
        }
        
        return PerformanceMetrics(
            accuracy=accuracy,
            precision=overall_precision,
            recall=overall_recall,
            f1_score=f1,
            confusion_matrix=cm,
            true_positives=int(tp),
            false_positives=int(fp),
            true_negatives=int(tn),
            false_negatives=int(fn),
            per_class_precision=per_class_precision,
            per_class_recall=per_class_recall,
            per_class_f1=per_class_f1,
            total_samples=len(labeled_results),
            class_distribution=class_distribution,
            confidence_stats=confidence_stats,
            classification_report=classification_report(y_true, y_pred, labels=self.class_names, zero_division=0),
            processing_stats=processing_stats
        )
    
    def _create_empty_metrics(self) -> PerformanceMetrics:
        """빈 성능 지표 생성"""
        return PerformanceMetrics(
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            confusion_matrix=np.array([]),
            true_positives=0,
            false_positives=0,
            true_negatives=0,
            false_negatives=0,
            per_class_precision={name: 0.0 for name in self.class_names},
            per_class_recall={name: 0.0 for name in self.class_names},
            per_class_f1={name: 0.0 for name in self.class_names},
            total_samples=0,
            class_distribution={},
            confidence_stats={},
            classification_report="",
            processing_stats={}
        )
    
    def save_results(self, save_detailed: bool = True, save_overlay: bool = True,
                    save_metrics: bool = True) -> Dict[str, str]:
        """모든 결과 저장
        
        Args:
            save_detailed: 상세 추론 결과 저장 여부
            save_overlay: 오버레이 데이터 저장 여부  
            save_metrics: 성능 지표 저장 여부
            
        Returns:
            저장된 파일 경로들
        """
        saved_files = {}
        
        # 1. 상세 추론 결과 저장
        if save_detailed and self.inference_results:
            detailed_file = self._save_detailed_results()
            saved_files['detailed_results'] = detailed_file
        
        # 2. 오버레이 데이터 저장
        if save_overlay and self.overlay_data_list:
            overlay_files = self._save_overlay_data()
            saved_files['overlay_data'] = overlay_files
        
        # 3. 성능 지표 저장
        if save_metrics:
            metrics = self.calculate_performance_metrics()
            metrics_files = self._save_performance_metrics(metrics)
            saved_files.update(metrics_files)
        
        return saved_files
    
    def _save_detailed_results(self) -> str:
        """상세 추론 결과 저장"""
        results_data = []
        
        for result in self.inference_results:
            result_dict = {
                'sample_id': result.sample_id,
                'true_label': result.true_label,
                'predicted_label': result.predicted_label,
                'confidence': result.confidence,
                'class_probabilities': result.class_probabilities,
                'processing_time': result.processing_time,
                'metadata': result.metadata
            }
            results_data.append(result_dict)
        
        # JSON으로 저장
        json_file = self.output_dir / "results" / "detailed_results.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        # CSV로도 저장 (간단한 버전)
        csv_data = []
        for result in self.inference_results:
            csv_row = {
                'sample_id': result.sample_id,
                'true_label': result.true_label,
                'predicted_label': result.predicted_label,
                'confidence': result.confidence,
                'processing_time': result.processing_time
            }
            csv_data.append(csv_row)
        
        csv_file = self.output_dir / "results" / "detailed_results.csv"
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        logging.info(f"Detailed results saved: {json_file}, {csv_file}")
        return str(json_file)
    
    def _save_overlay_data(self) -> List[str]:
        """오버레이 데이터 저장"""
        saved_files = []
        
        for overlay in self.overlay_data_list:
            # 개별 오버레이 데이터를 pkl로 저장 (레거시 코드 호환)
            overlay_dict = {
                'sample_id': overlay.sample_id,
                'video_path': overlay.video_path,
                'pose_data': overlay.pose_data,
                'tracking_data': overlay.tracking_data,
                'score_data': overlay.score_data,
                'window_results': [{
                    'window_id': result.window_id,
                    'predicted_class': result.predicted_class,
                    'confidence': result.confidence,
                    'class_probabilities': result.class_probabilities
                } for result in overlay.window_results],
                'final_prediction': overlay.final_prediction,
                'final_confidence': overlay.final_confidence,
                'frame_annotations': overlay.frame_annotations
            }
            
            pkl_file = self.output_dir / "overlay_data" / f"{overlay.sample_id}_overlay.pkl"
            with open(pkl_file, 'wb') as f:
                pickle.dump(overlay_dict, f)
            
            saved_files.append(str(pkl_file))
        
        logging.info(f"Overlay data saved: {len(saved_files)} files")
        return saved_files
    
    def _save_performance_metrics(self, metrics: PerformanceMetrics) -> Dict[str, str]:
        """성능 지표 저장"""
        saved_files = {}
        
        # 1. JSON 형식으로 성능 지표 저장
        metrics_dict = {
            'accuracy': metrics.accuracy,
            'precision': metrics.precision,
            'recall': metrics.recall,
            'f1_score': metrics.f1_score,
            'confusion_matrix': {
                'true_positives': metrics.true_positives,
                'false_positives': metrics.false_positives,
                'true_negatives': metrics.true_negatives,
                'false_negatives': metrics.false_negatives
            },
            'per_class_metrics': {
                'precision': metrics.per_class_precision,
                'recall': metrics.per_class_recall,
                'f1_score': metrics.per_class_f1
            },
            'total_samples': metrics.total_samples,
            'class_distribution': metrics.class_distribution,
            'confidence_stats': metrics.confidence_stats,
            'processing_stats': metrics.processing_stats,
            'classification_report': metrics.classification_report
        }
        
        json_file = self.output_dir / "performance_metrics" / "performance_metrics.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_dict, f, indent=2, ensure_ascii=False)
        
        saved_files['performance_metrics'] = str(json_file)
        
        # 2. 혼동 행렬을 별도 파일로 저장
        if metrics.confusion_matrix.size > 0:
            # numpy 배열로 저장
            cm_npy_file = self.output_dir / "confusion_matrices" / "confusion_matrix.npy"
            np.save(cm_npy_file, metrics.confusion_matrix)
            
            # CSV로도 저장
            cm_csv_file = self.output_dir / "confusion_matrices" / "confusion_matrix.csv"
            df_cm = pd.DataFrame(
                metrics.confusion_matrix, 
                index=self.class_names, 
                columns=self.class_names
            )
            df_cm.to_csv(cm_csv_file, encoding='utf-8')
            
            saved_files['confusion_matrix'] = str(cm_csv_file)
        
        # 3. PKL 형식으로 전체 메트릭 객체 저장
        pkl_file = self.output_dir / "performance_metrics" / "performance_metrics.pkl"
        with open(pkl_file, 'wb') as f:
            pickle.dump(metrics, f)
        
        saved_files['performance_metrics_pkl'] = str(pkl_file)
        
        logging.info(f"Performance metrics saved to {len(saved_files)} files")
        return saved_files
    
    def print_performance_summary(self, metrics: PerformanceMetrics = None):
        """성능 요약 출력"""
        if metrics is None:
            metrics = self.calculate_performance_metrics()
        
        print("=" * 70)
        print(" " * 25 + "PERFORMANCE SUMMARY")
        print("=" * 70)
        
        # 기본 성능 지표
        print(f"Accuracy:  {metrics.accuracy:.4f}")
        print(f"Precision: {metrics.precision:.4f}")
        print(f"Recall:    {metrics.recall:.4f}")
        print(f"F1-Score:  {metrics.f1_score:.4f}")
        
        print("-" * 70)
        
        # 혼동 행렬
        print("Confusion Matrix:")
        print(f"  True Positive:  {metrics.true_positives}")
        print(f"  False Positive: {metrics.false_positives}")
        print(f"  True Negative:  {metrics.true_negatives}")
        print(f"  False Negative: {metrics.false_negatives}")
        
        print("-" * 70)
        
        # 클래스별 성능
        print("Per-Class Performance:")
        for class_name in self.class_names:
            precision = metrics.per_class_precision.get(class_name, 0.0)
            recall = metrics.per_class_recall.get(class_name, 0.0)
            f1 = metrics.per_class_f1.get(class_name, 0.0)
            print(f"  {class_name:>10}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
        
        print("-" * 70)
        
        # 처리 성능
        print("Processing Performance:")
        stats = metrics.processing_stats
        print(f"  Total Samples: {stats.get('total_samples', 0)}")
        print(f"  Avg Time/Sample: {stats.get('avg_processing_time', 0.0):.4f}s")
        print(f"  FPS: {stats.get('fps', 0.0):.2f}")
        
        # 신뢰도 통계
        if metrics.confidence_stats:
            print(f"  Confidence: μ={metrics.confidence_stats.get('mean', 0.0):.3f}, "
                  f"σ={metrics.confidence_stats.get('std', 0.0):.3f}")
        
        print("=" * 70)
    
    def clear_results(self):
        """저장된 결과 초기화"""
        self.inference_results.clear()
        self.overlay_data_list.clear()
        logging.info("Results cleared")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """요약 통계 반환"""
        return {
            'total_inference_results': len(self.inference_results),
            'total_overlay_data': len(self.overlay_data_list),
            'labeled_results': len([r for r in self.inference_results if r.true_label is not None]),
            'class_names': self.class_names,
            'output_dir': str(self.output_dir)
        }