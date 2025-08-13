"""
결과 시각화 도구

추론 결과와 성능 분석을 시각화하는 도구입니다.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from typing import List, Dict, Any, Optional, Tuple
import seaborn as sns

from ..utils.data_structure import ClassificationResult
from ..pipelines.unified_pipeline import PipelineResult


class ResultVisualizer:
    """추론 결과 시각화 클래스"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        """
        Args:
            figsize: 그래프 크기
            dpi: 해상도
        """
        self.figsize = figsize
        self.dpi = dpi
        
        # 스타일 설정
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def visualize_classification_results(self, results: List[ClassificationResult], 
                                       save_path: Optional[str] = None) -> np.ndarray:
        """분류 결과 시각화
        
        Args:
            results: 분류 결과 리스트
            save_path: 저장 경로 (선택적)
            
        Returns:
            시각화 이미지
        """
        fig = Figure(figsize=self.figsize, dpi=self.dpi)
        
        # 2x2 서브플롯
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)
        
        # 1. 클래스별 분포
        class_counts = {}
        confidences_by_class = {}
        
        for result in results:
            class_name = result.predicted_class
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            if class_name not in confidences_by_class:
                confidences_by_class[class_name] = []
            confidences_by_class[class_name].append(result.confidence)
        
        # 클래스 분포 파이 차트
        if class_counts:
            ax1.pie(class_counts.values(), labels=class_counts.keys(), autopct='%1.1f%%')
            ax1.set_title('Class Distribution')
        
        # 2. 신뢰도 분포 히스토그램
        all_confidences = [r.confidence for r in results]
        if all_confidences:
            ax2.hist(all_confidences, bins=20, alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Confidence')
            ax2.set_ylabel('Count')
            ax2.set_title('Confidence Distribution')
            ax2.axvline(np.mean(all_confidences), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(all_confidences):.3f}')
            ax2.legend()
        
        # 3. 시간별 분류 결과
        window_ids = [r.window_id for r in results]
        fight_confidences = []
        nonfight_confidences = []
        
        for result in results:
            if 'Fight' in result.class_probabilities:
                fight_confidences.append(result.class_probabilities['Fight'])
            else:
                fight_confidences.append(0.0)
            
            if 'NonFight' in result.class_probabilities:
                nonfight_confidences.append(result.class_probabilities['NonFight'])
            else:
                nonfight_confidences.append(1.0 - fight_confidences[-1])
        
        if window_ids:
            ax3.plot(window_ids, fight_confidences, label='Fight', color='red', alpha=0.7)
            ax3.plot(window_ids, nonfight_confidences, label='NonFight', color='blue', alpha=0.7)
            ax3.set_xlabel('Window ID')
            ax3.set_ylabel('Probability')
            ax3.set_title('Classification Over Time')
            ax3.legend()
            ax3.set_ylim(0, 1)
        
        # 4. 클래스별 신뢰도 박스플롯
        if confidences_by_class:
            classes = list(confidences_by_class.keys())
            confidence_data = [confidences_by_class[cls] for cls in classes]
            
            ax4.boxplot(confidence_data, labels=classes)
            ax4.set_ylabel('Confidence')
            ax4.set_title('Confidence by Class')
        
        fig.tight_layout()
        
        # 이미지로 변환
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        
        # numpy 배열로 변환
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(int(height), int(width), 3)
        
        # BGR로 변환 (OpenCV 호환)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if save_path:
            cv2.imwrite(save_path, image)
        
        plt.close(fig)
        return image
    
    def visualize_pipeline_performance(self, result: PipelineResult, 
                                     save_path: Optional[str] = None) -> np.ndarray:
        """파이프라인 성능 분석 시각화
        
        Args:
            result: 파이프라인 결과
            save_path: 저장 경로 (선택적)
            
        Returns:
            성능 분석 이미지
        """
        fig = Figure(figsize=self.figsize, dpi=self.dpi)
        
        # 2x2 서브플롯
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)
        
        # 1. 단계별 처리 시간
        stages = ['Pose\nExtraction', 'Tracking', 'Scoring', 'Classification']
        times = [
            result.pose_extraction_time,
            result.tracking_time,
            result.scoring_time,
            result.classification_time
        ]
        
        colors = ['skyblue', 'lightgreen', 'orange', 'pink']
        bars = ax1.bar(stages, times, color=colors)
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Processing Time by Stage')
        ax1.tick_params(axis='x', rotation=45)
        
        # 시간 라벨 추가
        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time:.2f}s', ha='center', va='bottom')
        
        # 2. 시간 비율 파이 차트
        total_time = sum(times)
        if total_time > 0:
            percentages = [(t/total_time)*100 for t in times]
            ax2.pie(percentages, labels=stages, autopct='%1.1f%%', colors=colors)
            ax2.set_title('Time Distribution by Stage')
        
        # 3. 처리량 정보
        metrics = ['Total\nFrames', 'Processed\nWindows', 'Avg FPS', 'Processing\nTime (s)']
        values = [
            result.total_frames,
            result.processed_windows,
            result.avg_fps,
            result.processing_time
        ]
        
        bars = ax3.bar(metrics, values, color=['lightcoral', 'gold', 'lightblue', 'lightpink'])
        ax3.set_title('Overall Performance Metrics')
        ax3.tick_params(axis='x', rotation=45)
        
        # 값 라벨 추가
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}', ha='center', va='bottom')
        
        # 4. 분류 결과 요약
        classification_summary = self._analyze_classification_results(result.classification_results)
        
        if classification_summary:
            classes = list(classification_summary.keys())
            counts = [classification_summary[cls]['count'] for cls in classes]
            
            bars = ax4.bar(classes, counts, color=['red', 'blue'])
            ax4.set_ylabel('Count')
            ax4.set_title('Classification Results Summary')
            
            # 평균 신뢰도 표시
            for i, cls in enumerate(classes):
                avg_conf = classification_summary[cls]['avg_confidence']
                ax4.text(bars[i].get_x() + bars[i].get_width()/2.,
                        bars[i].get_height(),
                        f'Avg: {avg_conf:.2f}',
                        ha='center', va='bottom')
        
        fig.tight_layout()
        
        # 이미지로 변환
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(int(height), int(width), 3)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if save_path:
            cv2.imwrite(save_path, image)
        
        plt.close(fig)
        return image
    
    def create_confusion_matrix(self, results: List[ClassificationResult], 
                              true_labels: List[str],
                              save_path: Optional[str] = None) -> np.ndarray:
        """혼동 행렬 시각화
        
        Args:
            results: 분류 결과 리스트
            true_labels: 실제 라벨 리스트
            save_path: 저장 경로 (선택적)
            
        Returns:
            혼동 행렬 이미지
        """
        if len(results) != len(true_labels):
            raise ValueError("Results and true labels must have same length")
        
        # 예측값 추출
        predicted_labels = [r.predicted_class for r in results]
        
        # 고유 라벨
        unique_labels = sorted(list(set(true_labels + predicted_labels)))
        
        # 혼동 행렬 계산
        n_classes = len(unique_labels)
        confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
        
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        
        for true_label, pred_label in zip(true_labels, predicted_labels):
            true_idx = label_to_idx[true_label]
            pred_idx = label_to_idx[pred_label]
            confusion_matrix[true_idx, pred_idx] += 1
        
        # 시각화
        fig = Figure(figsize=(8, 6), dpi=self.dpi)
        ax = fig.add_subplot(1, 1, 1)
        
        # 히트맵
        im = ax.imshow(confusion_matrix, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        # 축 설정
        ax.set(xticks=np.arange(n_classes),
               yticks=np.arange(n_classes),
               xticklabels=unique_labels,
               yticklabels=unique_labels,
               title='Confusion Matrix',
               ylabel='True Label',
               xlabel='Predicted Label')
        
        # 텍스트 추가
        thresh = confusion_matrix.max() / 2.
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(j, i, format(confusion_matrix[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if confusion_matrix[i, j] > thresh else "black")
        
        fig.tight_layout()
        
        # 이미지로 변환
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(int(height), int(width), 3)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if save_path:
            cv2.imwrite(save_path, image)
        
        plt.close(fig)
        return image
    
    def _analyze_classification_results(self, results: List[ClassificationResult]) -> Dict[str, Dict]:
        """분류 결과 분석"""
        analysis = {}
        
        for result in results:
            class_name = result.predicted_class
            
            if class_name not in analysis:
                analysis[class_name] = {
                    'count': 0,
                    'confidences': [],
                    'avg_confidence': 0.0
                }
            
            analysis[class_name]['count'] += 1
            analysis[class_name]['confidences'].append(result.confidence)
        
        # 평균 신뢰도 계산
        for class_name in analysis:
            confidences = analysis[class_name]['confidences']
            analysis[class_name]['avg_confidence'] = np.mean(confidences) if confidences else 0.0
        
        return analysis
    
    def create_timeline_visualization(self, results: List[ClassificationResult], 
                                    video_fps: float = 30.0,
                                    save_path: Optional[str] = None) -> np.ndarray:
        """타임라인 시각화
        
        Args:
            results: 분류 결과 리스트
            video_fps: 비디오 FPS
            save_path: 저장 경로 (선택적)
            
        Returns:
            타임라인 이미지
        """
        if not results:
            return np.zeros((400, 800, 3), dtype=np.uint8)
        
        fig = Figure(figsize=(16, 6), dpi=self.dpi)
        ax = fig.add_subplot(1, 1, 1)
        
        # 시간 축 계산 (윈도우 ID를 시간으로 변환)
        times = [r.window_id / video_fps for r in results]  # 초 단위
        fight_probs = []
        
        for result in results:
            if 'Fight' in result.class_probabilities:
                fight_probs.append(result.class_probabilities['Fight'])
            else:
                fight_probs.append(1.0 if result.predicted_class == 'Fight' else 0.0)
        
        # 타임라인 그래프
        ax.plot(times, fight_probs, linewidth=2, alpha=0.8, label='Fight Probability')
        ax.fill_between(times, fight_probs, alpha=0.3)
        
        # 임계값 선
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Threshold (0.5)')
        
        # Fight 구간 강조
        fight_regions = []
        current_start = None
        
        for i, (time, prob) in enumerate(zip(times, fight_probs)):
            if prob > 0.5 and current_start is None:
                current_start = time
            elif prob <= 0.5 and current_start is not None:
                fight_regions.append((current_start, time))
                current_start = None
        
        # 마지막 구간 처리
        if current_start is not None:
            fight_regions.append((current_start, times[-1]))
        
        # Fight 구간 하이라이트
        for start_time, end_time in fight_regions:
            ax.axvspan(start_time, end_time, color='red', alpha=0.2, label='Fight Detection')
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Fight Probability')
        ax.set_title('Violence Detection Timeline')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 요약 통계 텍스트
        total_time = times[-1] - times[0] if len(times) > 1 else 0
        fight_time = sum(end - start for start, end in fight_regions)
        fight_percentage = (fight_time / total_time * 100) if total_time > 0 else 0
        
        stats_text = f'Total Duration: {total_time:.1f}s | Fight Time: {fight_time:.1f}s ({fight_percentage:.1f}%)'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        fig.tight_layout()
        
        # 이미지로 변환
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(int(height), int(width), 3)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if save_path:
            cv2.imwrite(save_path, image)
        
        plt.close(fig)
        return image