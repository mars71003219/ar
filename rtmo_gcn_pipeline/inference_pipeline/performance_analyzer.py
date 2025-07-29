#!/usr/bin/env python3
"""
Performance Analysis Module
성능 분석 모듈 - 파이프라인 각 단계별 성능 측정 및 bottleneck 분석
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class StageMetrics:
    """단계별 성능 메트릭"""
    stage_name: str
    total_time: float
    frame_count: int
    fps: float
    memory_peak: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            'stage_name': self.stage_name,
            'total_time': self.total_time,
            'frame_count': self.frame_count,
            'fps': self.fps,
            'memory_peak': self.memory_peak
        }

class PerformanceAnalyzer:
    """
    성능 분석기 - 파이프라인 각 단계별 처리 시간 측정
    """
    
    def __init__(self):
        self.stage_times = defaultdict(list)
        self.current_stage = None
        self.stage_start_time = None
        self.total_frames = 0
        self.memory_tracker = {}
        
    def start_stage(self, stage_name: str):
        """단계 시작"""
        if self.current_stage is not None:
            logger.warning(f"이전 단계 {self.current_stage}가 완료되지 않음")
            
        self.current_stage = stage_name
        self.stage_start_time = time.time()
        
        # GPU 메모리 사용량 추적 (가능한 경우)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                self.memory_tracker[stage_name + '_start'] = torch.cuda.memory_allocated() / 1024**2  # MB
        except Exception:
            pass
    
    def end_stage(self) -> float:
        """단계 종료 및 시간 반환"""
        if self.current_stage is None or self.stage_start_time is None:
            logger.warning("시작되지 않은 단계를 종료하려고 함")
            return 0.0
            
        elapsed_time = time.time() - self.stage_start_time
        self.stage_times[self.current_stage].append(elapsed_time)
        
        # GPU 메모리 사용량 추적 (가능한 경우)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                self.memory_tracker[self.current_stage + '_end'] = torch.cuda.memory_allocated() / 1024**2  # MB
        except Exception:
            pass
        
        logger.info(f"{self.current_stage} 완료: {elapsed_time:.3f}초")
        
        stage_name = self.current_stage
        self.current_stage = None
        self.stage_start_time = None
        
        return elapsed_time
    
    def get_stage_metrics(self, stage_name: str, frame_count: Optional[int] = None) -> StageMetrics:
        """단계별 메트릭 계산"""
        times = self.stage_times.get(stage_name, [])
        if not times:
            return StageMetrics(stage_name, 0.0, 0, 0.0)
            
        total_time = sum(times)
        avg_time = total_time / len(times)
        
        if frame_count is None:
            frame_count = self.total_frames
            
        fps = frame_count / total_time if total_time > 0 else 0.0
        
        # 메모리 피크 계산
        memory_peak = None
        start_mem = self.memory_tracker.get(stage_name + '_start')
        end_mem = self.memory_tracker.get(stage_name + '_end')
        if start_mem is not None and end_mem is not None:
            memory_peak = max(start_mem, end_mem)
        
        return StageMetrics(stage_name, total_time, frame_count, fps, memory_peak)
    
    def analyze_performance(self, total_frames: int) -> Dict:
        """전체 성능 분석"""
        self.total_frames = total_frames
        
        analysis = {
            'total_frames': total_frames,
            'stages': {},
            'bottlenecks': [],
            'recommendations': []
        }
        
        # 각 단계별 메트릭 계산
        stage_metrics = {}
        for stage_name in self.stage_times.keys():
            metrics = self.get_stage_metrics(stage_name, total_frames)
            stage_metrics[stage_name] = metrics
            analysis['stages'][stage_name] = metrics.to_dict()
        
        # Bottleneck 분석
        if stage_metrics:
            # 가장 시간이 오래 걸리는 단계들
            sorted_stages = sorted(stage_metrics.values(), key=lambda x: x.total_time, reverse=True)
            total_time = sum(m.total_time for m in stage_metrics.values())
            
            for metrics in sorted_stages[:3]:  # 상위 3개 단계
                percentage = (metrics.total_time / total_time) * 100 if total_time > 0 else 0
                if percentage > 20:  # 20% 이상이면 bottleneck
                    analysis['bottlenecks'].append({
                        'stage': metrics.stage_name,
                        'time': metrics.total_time,
                        'percentage': percentage,
                        'fps': metrics.fps
                    })
        
        # 최적화 권장사항
        analysis['recommendations'] = self._generate_recommendations(stage_metrics, total_frames)
        
        return analysis
    
    def _generate_recommendations(self, stage_metrics: Dict[str, StageMetrics], total_frames: int) -> List[str]:
        """최적화 권장사항 생성"""
        recommendations = []
        
        if not stage_metrics:
            return recommendations
        
        # 전체 FPS 계산
        total_time = sum(m.total_time for m in stage_metrics.values())
        overall_fps = total_frames / total_time if total_time > 0 else 0
        
        if overall_fps < 15:  # 15fps 미만이면 성능 개선 필요
            recommendations.append(f"전체 FPS {overall_fps:.1f}fps로 성능 개선 필요 (목표: 20-30fps)")
        
        # 단계별 분석 및 권장사항
        for stage_name, metrics in stage_metrics.items():
            if 'pose_estimation' in stage_name.lower():
                if metrics.fps < 20:
                    recommendations.append("포즈 추정 최적화: 배치 처리, 해상도 조정, 모델 최적화 고려")
                    
            elif 'tracking' in stage_name.lower():
                if metrics.fps < 50:
                    recommendations.append("트래킹 최적화: ByteTrack 설정 조정, detection threshold 최적화")
                    
            elif 'classification' in stage_name.lower():
                if metrics.fps < 100:
                    recommendations.append("분류 최적화: 배치 크기 증가, 시퀀스 길이 조정 고려")
                    
            elif 'overlay' in stage_name.lower():
                if metrics.fps < 30:
                    recommendations.append("오버레이 최적화: OpenCV 최적화, MMPose 시각화 설정 조정")
        
        return recommendations
    
    def save_analysis(self, analysis: Dict, output_path: str):
        """분석 결과 저장"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        logger.info(f"성능 분석 결과 저장: {output_path}")
    
    def print_summary(self, analysis: Dict):
        """분석 결과 요약 출력"""
        print("\n" + "="*60)
        print("🚀 파이프라인 성능 분석 결과")
        print("="*60)
        
        total_frames = analysis['total_frames']
        print(f"📹 총 프레임 수: {total_frames}")
        
        print("\n📊 단계별 성능:")
        for stage_name, metrics in analysis['stages'].items():
            print(f"  • {stage_name}:")
            print(f"    - 처리 시간: {metrics['total_time']:.2f}초")
            print(f"    - FPS: {metrics['fps']:.1f}")
            if metrics['memory_peak']:
                print(f"    - 메모리 피크: {metrics['memory_peak']:.1f}MB")
        
        # 전체 FPS 계산
        total_time = sum(m['total_time'] for m in analysis['stages'].values())
        overall_fps = total_frames / total_time if total_time > 0 else 0
        print(f"\n🎯 전체 성능: {overall_fps:.1f} FPS ({total_time:.2f}초)")
        
        if analysis['bottlenecks']:
            print(f"\n🔴 Bottleneck 단계:")
            for bottleneck in analysis['bottlenecks']:
                print(f"  • {bottleneck['stage']}: {bottleneck['percentage']:.1f}% ({bottleneck['time']:.2f}초)")
        
        if analysis['recommendations']:
            print(f"\n💡 최적화 권장사항:")
            for i, rec in enumerate(analysis['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        print("="*60)

# Context manager for easy stage timing
class StageTimer:
    """단계별 시간 측정용 컨텍스트 매니저"""
    
    def __init__(self, analyzer: PerformanceAnalyzer, stage_name: str):
        self.analyzer = analyzer
        self.stage_name = stage_name
    
    def __enter__(self):
        self.analyzer.start_stage(self.stage_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.analyzer.end_stage()