"""
Evaluation 결과 시각화 도구
혼동행렬, ROC 커브, PR 커브 등 평가 차트 생성
"""

import os
import json
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve, 
    average_precision_score, classification_report
)
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)

# 한글 폰트 설정 (컨테이너 환경 고려)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class EvaluationVisualizer:
    """평가 결과 시각화 클래스"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.charts_dir = self.output_dir / 'charts'
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        
        # 스타일 설정
        plt.style.use('default')
        sns.set_palette("husl")
    
    def generate_all_charts(self, summary_results: List[Dict], detailed_results: List[Dict], 
                          metrics: Dict[str, Any]) -> bool:
        """모든 차트 생성"""
        try:
            logger.info("Generating evaluation charts and tables")
            
            # 1. 혼동행렬 매트릭스
            self._create_confusion_matrix(metrics)
            
            # 2. 성능 지표 테이블
            self._create_metrics_table(metrics)
            
            # 3. ROC 커브
            self._create_roc_curve(summary_results)
            
            # 4. Precision-Recall 커브
            self._create_precision_recall_curve(summary_results)
            
            # 5. 비디오별 성능 차트
            self._create_video_performance_chart(summary_results)
            
            # 6. 윈도우별 점수 분포
            self._create_score_distribution(detailed_results)
            
            # 7. 클래스별 성능 비교
            self._create_class_performance_comparison(summary_results)
            
            logger.info(f"All charts saved to: {self.charts_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating charts: {e}")
            return False
    
    def _create_confusion_matrix(self, metrics: Dict[str, Any]):
        """혼동행렬 시각화"""
        cm = metrics['confusion_matrix']
        
        # 혼동행렬 데이터 준비
        matrix = np.array([[cm['TN'], cm['FP']], 
                          [cm['FN'], cm['TP']]])
        
        # 그래프 생성
        plt.figure(figsize=(8, 6))
        
        # 히트맵 생성
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['NonFight', 'Fight'],
                    yticklabels=['NonFight', 'Fight'],
                    cbar_kws={'label': 'Count'})
        
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        
        # 각 셀에 퍼센트 추가
        total = matrix.sum()
        for i in range(2):
            for j in range(2):
                percentage = matrix[i, j] / total * 100
                plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                        ha='center', va='center', fontsize=10, color='gray')
        
        plt.tight_layout()
        plt.savefig(self.charts_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Confusion matrix chart saved")
    
    def _create_metrics_table(self, metrics: Dict[str, Any]):
        """성능 지표 테이블 생성"""
        # 테이블 데이터 준비
        table_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity'],
            'Value': [
                f"{metrics['accuracy']:.4f}",
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{metrics['f1_score']:.4f}",
                f"{metrics['specificity']:.4f}"
            ],
            'Percentage': [
                f"{metrics['accuracy']*100:.2f}%",
                f"{metrics['precision']*100:.2f}%",
                f"{metrics['recall']*100:.2f}%",
                f"{metrics['f1_score']*100:.2f}%",
                f"{metrics['specificity']*100:.2f}%"
            ]
        }
        
        df = pd.DataFrame(table_data)
        
        # 테이블 시각화
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('tight')
        ax.axis('off')
        
        # 테이블 생성
        table = ax.table(cellText=df.values,
                        colLabels=df.columns,
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        # 테이블 스타일링
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2)
        
        # 헤더 스타일
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # 데이터 행 스타일
        for i in range(1, len(df) + 1):
            for j in range(len(df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F2F2F2')
        
        plt.title('Performance Metrics Summary', fontsize=16, fontweight='bold', pad=20)
        plt.savefig(self.charts_dir / 'metrics_table.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Metrics table saved")
    
    def _create_roc_curve(self, summary_results: List[Dict]):
        """ROC 커브 생성"""
        # 데이터 준비
        y_true = [1 if r['class_label'].lower() in ['fight', 'violence'] else 0 for r in summary_results]
        y_scores = []
        
        # 점수가 있는 경우와 없는 경우 처리
        for result in summary_results:
            if 'avg_confidence' in result:
                score = result['avg_confidence']
            else:
                # Fight 예측이면 높은 점수, NonFight 예측이면 낮은 점수 할당
                score = 0.7 if result['predicted_class'] == 'Fight' else 0.3
            y_scores.append(score)
        
        if len(set(y_true)) < 2:
            logger.warning("Cannot create ROC curve: only one class present")
            return
        
        # ROC 커브 계산
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # 그래프 생성
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.8)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.charts_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curve saved (AUC = {roc_auc:.3f})")
    
    def _create_precision_recall_curve(self, summary_results: List[Dict]):
        """Precision-Recall 커브 생성"""
        # 데이터 준비
        y_true = [1 if r['class_label'].lower() in ['fight', 'violence'] else 0 for r in summary_results]
        y_scores = []
        
        for result in summary_results:
            if 'avg_confidence' in result:
                score = result['avg_confidence']
            else:
                score = 0.7 if result['predicted_class'] == 'Fight' else 0.3
            y_scores.append(score)
        
        if len(set(y_true)) < 2 or sum(y_true) == 0:
            logger.warning("Cannot create PR curve: insufficient positive samples")
            return
        
        # PR 커브 계산
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        avg_precision = average_precision_score(y_true, y_scores)
        
        # 그래프 생성
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, 
                label=f'PR curve (mAP = {avg_precision:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.charts_dir / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Precision-Recall curve saved (mAP = {avg_precision:.3f})")
    
    def _create_video_performance_chart(self, summary_results: List[Dict]):
        """비디오별 성능 차트"""
        # 데이터 준비
        videos = [r['video_filename'] for r in summary_results]
        performance_types = [r['confusion_matrix'] for r in summary_results]
        classes = [r['class_label'] for r in summary_results]
        
        # 색상 매핑
        color_map = {'TP': 'green', 'TN': 'blue', 'FP': 'red', 'FN': 'orange'}
        colors = [color_map[pt] for pt in performance_types]
        
        # 그래프 생성
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(videos)), [1]*len(videos), color=colors, alpha=0.7)
        
        # 각 바에 라벨 추가
        for i, (video, perf_type, class_label) in enumerate(zip(videos, performance_types, classes)):
            plt.text(i, 0.5, f'{perf_type}\n{class_label}', 
                    ha='center', va='center', fontweight='bold', fontsize=10)
        
        plt.xlabel('Video Files', fontsize=12)
        plt.ylabel('Performance', fontsize=12)
        plt.title('Video-level Performance Results', fontsize=14, fontweight='bold')
        plt.xticks(range(len(videos)), videos, rotation=45, ha='right')
        
        # 범례 생성
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.7, label=perf_type) 
                          for perf_type, color in color_map.items()]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(self.charts_dir / 'video_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Video performance chart saved")
    
    def _create_score_distribution(self, detailed_results: List[Dict]):
        """윈도우별 점수 분포 차트"""
        if not detailed_results:
            return
        
        # 데이터 준비 - 비디오별로 분리
        video_scores = {}
        for result in detailed_results:
            video_name = result['video_filename']
            if video_name not in video_scores:
                video_scores[video_name] = []
            video_scores[video_name].append(result['fight_score'])
        
        # 그래프 생성
        plt.figure(figsize=(12, 6))
        
        # 각 비디오별로 히스토그램 생성
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for i, (video_name, scores) in enumerate(video_scores.items()):
            plt.hist(scores, bins=20, alpha=0.6, label=video_name, 
                    color=colors[i % len(colors)], density=True)
        
        plt.xlabel('Fight Score', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title('Fight Score Distribution by Video', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.charts_dir / 'score_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Score distribution chart saved")
    
    def _create_class_performance_comparison(self, summary_results: List[Dict]):
        """클래스별 성능 비교 차트"""
        # 클래스별 성능 계산
        fight_results = [r for r in summary_results if r['class_label'].lower() in ['fight', 'violence']]
        normal_results = [r for r in summary_results if r['class_label'].lower() in ['normal', 'nonfight']]
        
        # 정확도 계산
        fight_accuracy = sum(1 for r in fight_results if r['confusion_matrix'] in ['TP', 'TN']) / len(fight_results) if fight_results else 0
        normal_accuracy = sum(1 for r in normal_results if r['confusion_matrix'] in ['TP', 'TN']) / len(normal_results) if normal_results else 0
        
        # 그래프 생성
        plt.figure(figsize=(10, 6))
        
        classes = ['Fight', 'Normal']
        accuracies = [fight_accuracy, normal_accuracy]
        colors = ['red', 'blue']
        
        bars = plt.bar(classes, accuracies, color=colors, alpha=0.7)
        
        # 값 표시
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
        
        plt.ylim(0, 1.1)
        plt.ylabel('Accuracy', fontsize=12)
        plt.xlabel('Class', fontsize=12)
        plt.title('Class-wise Performance Comparison', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.charts_dir / 'class_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Class performance comparison saved")