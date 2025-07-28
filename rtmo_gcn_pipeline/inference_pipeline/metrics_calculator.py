#!/usr/bin/env python3
"""
Performance Metrics Calculator
성능 메트릭 계산 모듈 - TP/TN/FP/FN 기반 정확도, 정밀도, 재현율, F1-score 계산
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
import json
import os.path as osp

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """
    성능 메트릭 계산기
    예측 결과와 실제 라벨을 비교하여 분류 성능 지표 계산
    """
    
    def __init__(self):
        """초기화"""
        self.class_mapping = {
            0: 'NonFight',
            1: 'Fight'
        }
        
    def calculate_confusion_matrix(self, predictions: List[int], ground_truths: List[int]) -> Dict[str, int]:
        """
        혼동 행렬 계산
        
        Args:
            predictions: 예측 결과 리스트
            ground_truths: 실제 라벨 리스트
            
        Returns:
            혼동 행렬 딕셔너리 {'TP': int, 'TN': int, 'FP': int, 'FN': int}
        """
        if len(predictions) != len(ground_truths):
            logger.error(f"예측과 실제 라벨 개수가 다릅니다: {len(predictions)} vs {len(ground_truths)}")
            raise ValueError("예측과 실제 라벨 개수가 일치하지 않습니다")
        
        tp = sum(1 for p, gt in zip(predictions, ground_truths) if p == 1 and gt == 1)  # True Positive
        tn = sum(1 for p, gt in zip(predictions, ground_truths) if p == 0 and gt == 0)  # True Negative
        fp = sum(1 for p, gt in zip(predictions, ground_truths) if p == 1 and gt == 0)  # False Positive
        fn = sum(1 for p, gt in zip(predictions, ground_truths) if p == 0 and gt == 1)  # False Negative
        
        confusion_matrix = {
            'TP': tp,  # Fight를 Fight로 정확히 분류
            'TN': tn,  # NonFight를 NonFight로 정확히 분류  
            'FP': fp,  # NonFight를 Fight로 잘못 분류
            'FN': fn   # Fight를 NonFight로 잘못 분류
        }
        
        logger.info(f"혼동 행렬: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
        
        return confusion_matrix
    
    def calculate_metrics(self, confusion_matrix: Dict[str, int]) -> Dict[str, float]:
        """
        분류 성능 메트릭 계산
        
        Args:
            confusion_matrix: 혼동 행렬
            
        Returns:
            성능 메트릭 딕셔너리
        """
        tp = confusion_matrix['TP']
        tn = confusion_matrix['TN']
        fp = confusion_matrix['FP']
        fn = confusion_matrix['FN']
        
        total = tp + tn + fp + fn
        
        # 정확도 (Accuracy): 전체 중 올바르게 분류한 비율
        accuracy = (tp + tn) / total if total > 0 else 0.0
        
        # 정밀도 (Precision): Fight로 예측한 것 중 실제 Fight인 비율
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # 재현율 (Recall/Sensitivity): 실제 Fight 중 올바르게 Fight로 예측한 비율
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # 특이도 (Specificity): 실제 NonFight 중 올바르게 NonFight로 예측한 비율
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # F1-Score: 정밀도와 재현율의 조화평균
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # False Positive Rate: 실제 NonFight 중 Fight로 잘못 예측한 비율
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # False Negative Rate: 실제 Fight 중 NonFight로 잘못 예측한 비율
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1_score,
            'false_positive_rate': fpr,
            'false_negative_rate': fnr
        }
        
        logger.info(f"성능 메트릭: 정확도={accuracy:.4f}, 정밀도={precision:.4f}, 재현율={recall:.4f}, F1={f1_score:.4f}")
        
        return metrics
    
    def analyze_predictions_by_class(self, predictions: List[int], ground_truths: List[int], 
                                   confidences: Optional[List[float]] = None) -> Dict:
        """
        클래스별 예측 분석
        
        Args:
            predictions: 예측 결과
            ground_truths: 실제 라벨
            confidences: 예측 신뢰도 (선택사항)
            
        Returns:
            클래스별 분석 결과
        """
        analysis = {}
        
        for class_idx, class_name in self.class_mapping.items():
            # 해당 클래스의 실제 샘플들
            true_indices = [i for i, gt in enumerate(ground_truths) if gt == class_idx]
            # 해당 클래스로 예측된 샘플들
            pred_indices = [i for i, pred in enumerate(predictions) if pred == class_idx]
            
            # 올바르게 분류된 샘플들
            correct_indices = [i for i in true_indices if predictions[i] == class_idx]
            
            class_analysis = {
                'total_true_samples': len(true_indices),
                'total_predicted_samples': len(pred_indices),
                'correctly_classified': len(correct_indices),
                'class_accuracy': len(correct_indices) / len(true_indices) if true_indices else 0.0
            }
            
            # 신뢰도 분석 (제공된 경우)
            if confidences:
                if true_indices:
                    true_confidences = [confidences[i] for i in true_indices]
                    class_analysis['avg_confidence_true'] = np.mean(true_confidences)
                    class_analysis['std_confidence_true'] = np.std(true_confidences)
                
                if correct_indices:
                    correct_confidences = [confidences[i] for i in correct_indices]
                    class_analysis['avg_confidence_correct'] = np.mean(correct_confidences)
            
            analysis[class_name] = class_analysis
        
        return analysis
    
    def calculate_comprehensive_metrics(self, predictions: List[int], ground_truths: List[int],
                                      confidences: Optional[List[float]] = None,
                                      video_names: Optional[List[str]] = None) -> Dict:
        """
        종합적인 성능 메트릭 계산
        
        Args:
            predictions: 예측 결과
            ground_truths: 실제 라벨
            confidences: 예측 신뢰도
            video_names: 비디오 이름 리스트
            
        Returns:
            종합 메트릭 결과
        """
        # 기본 혼동 행렬 및 메트릭
        confusion_matrix = self.calculate_confusion_matrix(predictions, ground_truths)
        metrics = self.calculate_metrics(confusion_matrix)
        
        # 클래스별 분석
        class_analysis = self.analyze_predictions_by_class(predictions, ground_truths, confidences)
        
        # 전체 통계
        total_samples = len(predictions)
        correct_predictions = sum(1 for p, gt in zip(predictions, ground_truths) if p == gt)
        
        # 신뢰도 분석
        confidence_analysis = {}
        if confidences:
            confidence_analysis = {
                'mean_confidence': np.mean(confidences),
                'std_confidence': np.std(confidences),
                'min_confidence': np.min(confidences),
                'max_confidence': np.max(confidences),
                'confidence_correct': np.mean([confidences[i] for i in range(len(confidences)) 
                                              if predictions[i] == ground_truths[i]]),
                'confidence_incorrect': np.mean([confidences[i] for i in range(len(confidences)) 
                                                if predictions[i] != ground_truths[i]]) if any(predictions[i] != ground_truths[i] for i in range(len(predictions))) else 0.0
            }
        
        # 오분류 사례 분석
        misclassified_cases = []
        for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            if pred != gt:
                case = {
                    'index': i,
                    'predicted': self.class_mapping[pred],
                    'ground_truth': self.class_mapping[gt],
                    'confidence': confidences[i] if confidences else None,
                    'video_name': video_names[i] if video_names else f"video_{i}"
                }
                misclassified_cases.append(case)
        
        comprehensive_results = {
            'confusion_matrix': confusion_matrix,
            'metrics': metrics,
            'class_analysis': class_analysis,
            'confidence_analysis': confidence_analysis,
            'summary': {
                'total_samples': total_samples,
                'correct_predictions': correct_predictions,
                'incorrect_predictions': total_samples - correct_predictions,
                'overall_accuracy': correct_predictions / total_samples if total_samples > 0 else 0.0
            },
            'misclassified_cases': misclassified_cases
        }
        
        return comprehensive_results
    
    def compare_with_baseline(self, current_results: Dict, baseline_results: Dict) -> Dict:
        """
        기준선과 성능 비교
        
        Args:
            current_results: 현재 시스템 결과
            baseline_results: 기준 시스템 결과
            
        Returns:
            비교 결과
        """
        current_metrics = current_results['metrics']
        baseline_metrics = baseline_results['metrics']
        
        improvements = {}
        for metric_name in current_metrics:
            if metric_name in baseline_metrics:
                current_val = current_metrics[metric_name]
                baseline_val = baseline_metrics[metric_name]
                improvement = current_val - baseline_val
                improvement_pct = (improvement / baseline_val * 100) if baseline_val > 0 else 0
                
                improvements[metric_name] = {
                    'current': current_val,
                    'baseline': baseline_val,
                    'improvement': improvement,
                    'improvement_percentage': improvement_pct
                }
        
        # 혼동 행렬 비교
        cm_comparison = {}
        current_cm = current_results['confusion_matrix']
        baseline_cm = baseline_results['confusion_matrix']
        
        for cm_key in current_cm:
            if cm_key in baseline_cm:
                cm_comparison[cm_key] = {
                    'current': current_cm[cm_key],
                    'baseline': baseline_cm[cm_key],
                    'change': current_cm[cm_key] - baseline_cm[cm_key]
                }
        
        return {
            'metric_improvements': improvements,
            'confusion_matrix_comparison': cm_comparison,
            'summary': {
                'better_metrics': [m for m, data in improvements.items() if data['improvement'] > 0],
                'worse_metrics': [m for m, data in improvements.items() if data['improvement'] < 0],
                'overall_improvement': np.mean([data['improvement'] for data in improvements.values()])
            }
        }
    
    def save_results(self, results: Dict, output_path: str):
        """결과를 JSON 파일로 저장"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"메트릭 결과 저장 완료: {output_path}")
        except Exception as e:
            logger.error(f"결과 저장 실패: {e}")
    
    def generate_report(self, results: Dict, output_path: str):
        """상세 분석 보고서 생성"""
        confusion_matrix = results['confusion_matrix']
        metrics = results['metrics']
        class_analysis = results['class_analysis']
        summary = results['summary']
        
        report = f"""# Violence Detection Performance Report

## 📊 전체 성능 요약

- **총 샘플 수**: {summary['total_samples']}
- **올바른 예측**: {summary['correct_predictions']}
- **잘못된 예측**: {summary['incorrect_predictions']}
- **전체 정확도**: {summary['overall_accuracy']:.4f}

## 🎯 성능 메트릭

| 메트릭 | 값 |
|--------|-----|
| 정확도 (Accuracy) | {metrics['accuracy']:.4f} |
| 정밀도 (Precision) | {metrics['precision']:.4f} |
| 재현율 (Recall) | {metrics['recall']:.4f} |
| 특이도 (Specificity) | {metrics['specificity']:.4f} |
| F1-Score | {metrics['f1_score']:.4f} |
| False Positive Rate | {metrics['false_positive_rate']:.4f} |
| False Negative Rate | {metrics['false_negative_rate']:.4f} |

## 📈 혼동 행렬

|          | 예측 NonFight | 예측 Fight |
|----------|---------------|------------|
| 실제 NonFight | {confusion_matrix['TN']} (TN) | {confusion_matrix['FP']} (FP) |
| 실제 Fight | {confusion_matrix['FN']} (FN) | {confusion_matrix['TP']} (TP) |

## 🔍 클래스별 분석

### Fight 클래스
- 실제 샘플 수: {class_analysis['Fight']['total_true_samples']}
- 예측된 샘플 수: {class_analysis['Fight']['total_predicted_samples']}
- 올바르게 분류된 수: {class_analysis['Fight']['correctly_classified']}
- 클래스 정확도: {class_analysis['Fight']['class_accuracy']:.4f}

### NonFight 클래스
- 실제 샘플 수: {class_analysis['NonFight']['total_true_samples']}
- 예측된 샘플 수: {class_analysis['NonFight']['total_predicted_samples']}
- 올바르게 분류된 수: {class_analysis['NonFight']['correctly_classified']}
- 클래스 정확도: {class_analysis['NonFight']['class_accuracy']:.4f}

## 📝 결론

이 보고서는 STGCN++ Fight 검출 시스템의 성능을 종합적으로 분석한 결과입니다.
"""

        # 신뢰도 분석 추가 (있는 경우)
        if 'confidence_analysis' in results and results['confidence_analysis']:
            conf = results['confidence_analysis']
            report += f"""
## 🔍 신뢰도 분석

- **평균 신뢰도**: {conf['mean_confidence']:.4f}
- **신뢰도 표준편차**: {conf['std_confidence']:.4f}
- **최소 신뢰도**: {conf['min_confidence']:.4f}
- **최대 신뢰도**: {conf['max_confidence']:.4f}
- **올바른 예측 평균 신뢰도**: {conf['confidence_correct']:.4f}
- **틀린 예측 평균 신뢰도**: {conf['confidence_incorrect']:.4f}
"""

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"성능 보고서 생성 완료: {output_path}")
        except Exception as e:
            logger.error(f"보고서 생성 실패: {e}")