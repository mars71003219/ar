#!/usr/bin/env python3
"""
Realtime Decision Engine - 실시간 의사결정 엔진

비디오 파일과 달리 실시간 스트림에서는 전체를 볼 수 없으므로
슬라이딩 윈도우 기반의 적응적 판단 로직을 사용합니다.
"""

import time
import numpy as np
from collections import deque
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .realtime_window_manager import WindowData


class AlertLevel(Enum):
    """알림 레벨"""
    NORMAL = "normal"
    SUSPICIOUS = "suspicious"  # 의심스러운 활동
    WARNING = "warning"        # 경고 (연속 탐지)
    CRITICAL = "critical"      # 긴급 (확정적 폭력)
    COOLING_DOWN = "cooling_down"  # 해제 중 (단계적 해제)


@dataclass
class DecisionResult:
    """의사결정 결과"""
    is_fight: bool
    alert_level: AlertLevel
    confidence: float
    consecutive_count: int
    recent_fight_ratio: float
    temporal_context: Dict[str, Any]
    reason: str
    
    # 알람 해제 관련 정보
    recovery_progress: float  # 회복 진행도 (0.0~1.0)
    stability_score: float    # 안정성 점수 (0.0~1.0)
    time_to_normal: Optional[int]  # NORMAL까지 예상 시간(초)


class RealtimeDecisionEngine:
    """
    실시간 의사결정 엔진
    
    기존 비디오 처리의 두 가지 조건을 실시간에 적응:
    1. 연속 이벤트 수 → 실시간 연속 카운팅
    2. Fight ratio → 최근 N개 윈도우의 슬라이딩 비율
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        초기화
        
        Args:
            config: 의사결정 설정
                {
                    'consecutive_threshold': 3,      # 연속 Fight 윈도우 임계값
                    'fight_ratio_threshold': 0.4,    # Fight 비율 임계값
                    'sliding_window_size': 20,       # 슬라이딩 윈도우 크기
                    'cooldown_period': 30,           # 쿨다운 기간 (초)
                    'confidence_decay': 0.1,         # 신뢰도 감쇄율
                    'temporal_weight': 0.8,          # 시간적 가중치
                }
        """
        self.config = config or {}
        
        # 설정값 초기화
        self.consecutive_threshold = self.config.get('consecutive_threshold', 3)
        self.fight_ratio_threshold = self.config.get('fight_ratio_threshold', 0.4)
        self.sliding_window_size = self.config.get('sliding_window_size', 20)
        self.cooldown_period = self.config.get('cooldown_period', 30)  # 초
        self.confidence_decay = self.config.get('confidence_decay', 0.1)
        self.temporal_weight = self.config.get('temporal_weight', 0.8)
        
        # 상태 관리
        self.window_history = deque(maxlen=self.sliding_window_size)
        self.consecutive_fight_count = 0
        self.last_fight_time = 0
        self.last_alert_time = 0
        
        # 알람 해제 관련 상태
        self.current_alert_level = AlertLevel.NORMAL
        self.alert_start_time = 0
        self.consecutive_normal_count = 0
        self.recovery_start_time = 0
        
        # 해제 조건 설정
        self.min_recovery_time = self.config.get('min_recovery_time', 15)  # 최소 회복 시간
        self.stability_threshold = self.config.get('stability_threshold', 0.8)  # 안정성 임계값
        self.normal_streak_required = self.config.get('normal_streak_required', 5)  # 연속 정상 윈도우 수
        
        # 시간적 컨텍스트
        self.fight_timestamps = deque(maxlen=100)  # 최근 Fight 타임스탬프
        self.alert_history = deque(maxlen=50)      # 알림 이력
        
        # 적응적 임계값
        self.adaptive_threshold = self.config.get('classification_threshold', 0.5)
        self.threshold_adjustment_factor = 0.05
        
        print(f"RealtimeDecisionEngine initialized with:")
        print(f"  consecutive_threshold: {self.consecutive_threshold}")
        print(f"  fight_ratio_threshold: {self.fight_ratio_threshold}")
        print(f"  sliding_window_size: {self.sliding_window_size}")
        print(f"  cooldown_period: {self.cooldown_period}s")
    
    def process_window(self, window_data: WindowData, 
                      prediction_result: Dict[str, Any]) -> DecisionResult:
        """
        윈도우 처리 및 실시간 의사결정
        
        Args:
            window_data: 윈도우 데이터
            prediction_result: STGCN++ 예측 결과
                {
                    'fight_score': float,
                    'is_fight': bool,
                    'confidence': float
                }
        
        Returns:
            DecisionResult: 의사결정 결과
        """
        current_time = time.time()
        
        # 1. 윈도우 이력에 추가
        window_record = {
            'window_idx': window_data.window_idx,
            'timestamp': current_time,
            'is_fight': prediction_result['is_fight'],
            'fight_score': prediction_result['fight_score'],
            'confidence': prediction_result['confidence'],
            'start_frame': window_data.start_frame,
            'end_frame': window_data.end_frame
        }
        
        self.window_history.append(window_record)
        
        # 2. 연속 Fight 카운팅 업데이트
        self._update_consecutive_count(prediction_result['is_fight'])
        
        # 3. 최근 Fight ratio 계산
        recent_fight_ratio = self._calculate_recent_fight_ratio()
        
        # 4. 시간적 컨텍스트 분석
        temporal_context = self._analyze_temporal_context(current_time)
        
        # 5. 적응적 임계값 조정
        self._adjust_adaptive_threshold(recent_fight_ratio)
        
        # 6. 최종 의사결정
        decision = self._make_final_decision(
            window_record=window_record,
            consecutive_count=self.consecutive_fight_count,
            recent_fight_ratio=recent_fight_ratio,
            temporal_context=temporal_context,
            current_time=current_time
        )
        
        # 7. Fight 이벤트 기록
        if decision.is_fight:
            self.fight_timestamps.append(current_time)
            self.last_fight_time = current_time
        
        # 8. 알림 이력 업데이트
        if decision.alert_level != AlertLevel.NORMAL:
            self.alert_history.append({
                'timestamp': current_time,
                'alert_level': decision.alert_level.value,
                'confidence': decision.confidence,
                'reason': decision.reason
            })
            self.last_alert_time = current_time
        
        return decision
    
    def _update_consecutive_count(self, is_fight: bool):
        """연속 Fight 카운트 및 정상 카운트 업데이트"""
        if is_fight:
            self.consecutive_fight_count += 1
            self.consecutive_normal_count = 0
        else:
            self.consecutive_fight_count = 0
            self.consecutive_normal_count += 1
    
    def _calculate_recent_fight_ratio(self) -> float:
        """최근 N개 윈도우의 Fight 비율 계산"""
        if not self.window_history:
            return 0.0
        
        fight_count = sum(1 for w in self.window_history if w['is_fight'])
        return fight_count / len(self.window_history)
    
    def _analyze_temporal_context(self, current_time: float) -> Dict[str, Any]:
        """시간적 컨텍스트 분석"""
        # 최근 Fight 밀도 계산
        recent_fights = [t for t in self.fight_timestamps 
                        if current_time - t <= 60]  # 최근 1분
        fight_density = len(recent_fights) / 60.0  # fights per second
        
        # 최근 알림 빈도
        recent_alerts = [a for a in self.alert_history 
                        if current_time - a['timestamp'] <= 300]  # 최근 5분
        alert_frequency = len(recent_alerts) / 300.0  # alerts per second
        
        # 쿨다운 상태 확인
        time_since_last_fight = current_time - self.last_fight_time
        is_cooldown = time_since_last_fight < self.cooldown_period
        
        # 시간대별 가중치 (예: 야간에 더 민감하게)
        hour = time.localtime(current_time).tm_hour
        time_weight = 1.2 if 22 <= hour or hour <= 6 else 1.0
        
        return {
            'fight_density': fight_density,
            'alert_frequency': alert_frequency,
            'time_since_last_fight': time_since_last_fight,
            'is_cooldown': is_cooldown,
            'time_weight': time_weight,
            'recent_fight_count': len(recent_fights),
            'recent_alert_count': len(recent_alerts)
        }
    
    def _adjust_adaptive_threshold(self, recent_fight_ratio: float):
        """적응적 임계값 조정"""
        # Fight 비율이 높으면 임계값을 약간 높여 false positive 줄임
        if recent_fight_ratio > 0.6:
            adjustment = self.threshold_adjustment_factor
        elif recent_fight_ratio < 0.2:
            adjustment = -self.threshold_adjustment_factor
        else:
            adjustment = 0
        
        self.adaptive_threshold = max(0.3, min(0.8, 
            self.adaptive_threshold + adjustment))
    
    def _make_final_decision(self, 
                           window_record: Dict[str, Any],
                           consecutive_count: int,
                           recent_fight_ratio: float,
                           temporal_context: Dict[str, Any],
                           current_time: float) -> DecisionResult:
        """최종 의사결정"""
        
        is_fight = window_record['is_fight']
        base_confidence = window_record['confidence']
        
        # 1. 기본 조건 확인 (기존 로직 적응)
        meets_consecutive = consecutive_count >= self.consecutive_threshold
        meets_ratio = recent_fight_ratio >= self.fight_ratio_threshold
        
        # 2. 시간적 컨텍스트 반영
        time_weight = temporal_context['time_weight']
        is_cooldown = temporal_context['is_cooldown']
        fight_density = temporal_context['fight_density']
        
        # 3. 신뢰도 계산 (시간적 가중치 적용)
        temporal_confidence = base_confidence * time_weight
        
        # Fight 밀도가 높으면 신뢰도 증가
        if fight_density > 0.05:  # 1분에 3번 이상
            temporal_confidence *= 1.2
        
        # 쿨다운 중이면 신뢰도 감소
        if is_cooldown:
            temporal_confidence *= (1 - self.confidence_decay)
        
        temporal_confidence = min(1.0, temporal_confidence)
        
        # 4. 알림 레벨 결정
        alert_level = self._determine_alert_level(
            is_fight=is_fight,
            consecutive_count=consecutive_count,
            recent_fight_ratio=recent_fight_ratio,
            temporal_confidence=temporal_confidence,
            fight_density=fight_density,
            is_cooldown=is_cooldown
        )
        
        # 5. 최종 Fight 판단
        final_is_fight = self._determine_final_fight_status(
            meets_consecutive=meets_consecutive,
            meets_ratio=meets_ratio,
            alert_level=alert_level,
            temporal_confidence=temporal_confidence
        )
        
        # 6. 알람 해제 분석
        recovery_info = self._analyze_alert_recovery(
            current_alert_level=alert_level,
            current_time=current_time,
            recent_fight_ratio=recent_fight_ratio,
            consecutive_normal_count=self.consecutive_normal_count,
            temporal_confidence=temporal_confidence
        )
        
        # 7. 판단 근거 생성
        reason = self._generate_decision_reason(
            is_fight=is_fight,
            consecutive_count=consecutive_count,
            recent_fight_ratio=recent_fight_ratio,
            meets_consecutive=meets_consecutive,
            meets_ratio=meets_ratio,
            final_is_fight=final_is_fight,
            alert_level=alert_level,
            recovery_info=recovery_info
        )
        
        return DecisionResult(
            is_fight=final_is_fight,
            alert_level=alert_level,
            confidence=temporal_confidence,
            consecutive_count=consecutive_count,
            recent_fight_ratio=recent_fight_ratio,
            temporal_context=temporal_context,
            reason=reason,
            recovery_progress=recovery_info['progress'],
            stability_score=recovery_info['stability'],
            time_to_normal=recovery_info['time_to_normal']
        )
    
    def _determine_alert_level(self, is_fight: bool, consecutive_count: int,
                              recent_fight_ratio: float, temporal_confidence: float,
                              fight_density: float, is_cooldown: bool) -> AlertLevel:
        """알림 레벨 결정"""
        
        if is_cooldown:
            return AlertLevel.NORMAL
        
        # 긴급 상황 (확정적 폭력)
        if (consecutive_count >= self.consecutive_threshold + 2 and 
            recent_fight_ratio >= 0.6 and 
            temporal_confidence >= 0.8):
            return AlertLevel.CRITICAL
        
        # 경고 (연속 탐지)
        if consecutive_count >= self.consecutive_threshold:
            return AlertLevel.WARNING
        
        # 의심스러운 활동
        if (is_fight and temporal_confidence >= 0.6) or fight_density > 0.03:
            return AlertLevel.SUSPICIOUS
        
        return self._determine_alert_level_with_recovery(
            base_level=base_level,
            current_time=current_time
        )
    
    def _determine_alert_level_with_recovery(self, base_level: AlertLevel, 
                                           current_time: float) -> AlertLevel:
        """알람 해제 로직을 포함한 최종 알람 레벨 결정"""
        
        # 현재 레벨이 NORMAL이면 그대로 유지
        if base_level == AlertLevel.NORMAL:
            self.current_alert_level = AlertLevel.NORMAL
            return AlertLevel.NORMAL
        
        # 알람이 새로 발생한 경우
        if self.current_alert_level == AlertLevel.NORMAL and base_level != AlertLevel.NORMAL:
            self.current_alert_level = base_level
            self.alert_start_time = current_time
            self.recovery_start_time = 0
            return base_level
        
        # 알람이 지속되는 경우 - 레벨 상승 가능
        if base_level.value in ['critical', 'warning'] and base_level != self.current_alert_level:
            self.current_alert_level = base_level
            return base_level
        
        # 알람 해제 조건 확인
        if self._should_start_recovery(base_level, current_time):
            if self.recovery_start_time == 0:
                self.recovery_start_time = current_time
            
            recovery_level = self._get_recovery_level(current_time)
            self.current_alert_level = recovery_level
            return recovery_level
        
        # 알람이 다시 악화된 경우
        if base_level.value in ['critical', 'warning'] and self.recovery_start_time > 0:
            self.recovery_start_time = 0  # 회복 프로세스 중단
            self.current_alert_level = base_level
            return base_level
        
        return self.current_alert_level
    
    def _should_start_recovery(self, base_level: AlertLevel, current_time: float) -> bool:
        """회복 프로세스 시작 조건 확인"""
        
        # 기본 레벨이 NORMAL이거나 SUSPICIOUS면 회복 가능
        if base_level not in [AlertLevel.NORMAL, AlertLevel.SUSPICIOUS]:
            return False
        
        # 최소 알람 지속 시간 확인
        if current_time - self.alert_start_time < self.min_recovery_time:
            return False
        
        # 연속 정상 윈도우 수 확인
        if self.consecutive_normal_count < 3:  # 최소 3개 정상 윈도우
            return False
        
        # Fight 비율이 충분히 낮아졌는지 확인
        recent_ratio = self._calculate_recent_fight_ratio()
        if recent_ratio > self.fight_ratio_threshold * 0.6:  # 원래 임계값의 60% 이하
            return False
        
        return True
    
    def _get_recovery_level(self, current_time: float) -> AlertLevel:
        """회복 단계에 따른 알람 레벨 반환"""
        recovery_time = current_time - self.recovery_start_time
        
        # 단계적 해제
        if recovery_time < self.min_recovery_time // 2:
            return AlertLevel.COOLING_DOWN
        elif recovery_time < self.min_recovery_time:
            # 추가 안정성 확인
            if self._is_situation_stable():
                return AlertLevel.NORMAL
            else:
                return AlertLevel.COOLING_DOWN
        else:
            # 충분한 시간 경과, 강제 해제
            return AlertLevel.NORMAL
    
    def _is_situation_stable(self) -> bool:
        """상황 안정성 확인"""
        
        # 연속 정상 윈도우 수 확인
        if self.consecutive_normal_count < self.normal_streak_required:
            return False
        
        # 최근 Fight 비율 확인
        recent_ratio = self._calculate_recent_fight_ratio()
        if recent_ratio > self.fight_ratio_threshold * 0.5:  # 원래 임계값의 50% 이하
            return False
        
        # 최근 Fight 밀도 확인
        current_time = time.time()
        recent_fights = [t for t in self.fight_timestamps 
                        if current_time - t <= 120]  # 최근 2분
        if len(recent_fights) > 2:  # 2분에 3번 이상이면 불안정
            return False
        
        return True
    
    def _analyze_alert_recovery(self, current_alert_level: AlertLevel,
                               current_time: float, recent_fight_ratio: float,
                               consecutive_normal_count: int,
                               temporal_confidence: float) -> Dict[str, Any]:
        """알람 해제 진행상황 분석"""
        
        if current_alert_level == AlertLevel.NORMAL:
            return {
                'progress': 1.0,
                'stability': 1.0,
                'time_to_normal': 0
            }
        
        # 회복 진행도 계산
        progress = 0.0
        if self.recovery_start_time > 0:
            recovery_time = current_time - self.recovery_start_time
            progress = min(1.0, recovery_time / self.min_recovery_time)
        
        # 안정성 점수 계산 (여러 요소 종합)
        stability_factors = {
            'normal_streak': min(1.0, consecutive_normal_count / self.normal_streak_required),
            'low_fight_ratio': max(0.0, 1.0 - (recent_fight_ratio / self.fight_ratio_threshold)),
            'time_stability': min(1.0, (current_time - self.last_fight_time) / self.cooldown_period),
            'confidence_stability': 1.0 - temporal_confidence if not is_fight else 0.0
        }
        
        stability = sum(stability_factors.values()) / len(stability_factors)
        
        # NORMAL까지 예상 시간 계산
        time_to_normal = None
        if current_alert_level != AlertLevel.NORMAL:
            remaining_normal_windows = max(0, self.normal_streak_required - consecutive_normal_count)
            time_to_normal = max(0, remaining_normal_windows * 10)  # 윈도우당 약 10초 가정
            
            if self.recovery_start_time == 0:
                time_to_normal += self.min_recovery_time
        
        return {
            'progress': progress,
            'stability': stability,
            'time_to_normal': time_to_normal,
            'factors': stability_factors
        }
    
    def _determine_final_fight_status(self, meets_consecutive: bool, 
                                    meets_ratio: bool, alert_level: AlertLevel,
                                    temporal_confidence: float) -> bool:
        """최종 Fight 상태 결정"""
        
        # 긴급 상황은 무조건 Fight
        if alert_level == AlertLevel.CRITICAL:
            return True
        
        # 기존 조건을 실시간에 적응
        # 연속 조건 OR 비율 조건 중 하나만 만족해도 Fight (더 민감하게)
        basic_fight = meets_consecutive or (meets_ratio and temporal_confidence >= 0.7)
        
        # 경고 레벨 이상이고 기본 조건 만족
        return (alert_level.value in ['warning', 'critical']) and basic_fight
    
    def _generate_decision_reason(self, is_fight: bool, consecutive_count: int,
                                recent_fight_ratio: float, meets_consecutive: bool,
                                meets_ratio: bool, final_is_fight: bool,
                                alert_level: AlertLevel, recovery_info: Dict[str, Any]) -> str:
        """판단 근거 생성"""
        reasons = []
        
        if final_is_fight:
            if consecutive_count >= self.consecutive_threshold:
                reasons.append(f"연속 {consecutive_count}회 Fight 탐지")
            
            if meets_ratio:
                reasons.append(f"최근 Fight 비율 {recent_fight_ratio:.2f}")
            
            if alert_level == AlertLevel.CRITICAL:
                reasons.append("긴급 상황 확정")
            elif alert_level == AlertLevel.WARNING:
                reasons.append("경고 레벨 도달")
        else:
            if not meets_consecutive:
                reasons.append(f"연속 조건 미달 ({consecutive_count}/{self.consecutive_threshold})")
            
            if not meets_ratio:
                reasons.append(f"비율 조건 미달 ({recent_fight_ratio:.2f}/{self.fight_ratio_threshold})")
        
        # 알람 해제 관련 정보 추가
        if alert_level == AlertLevel.COOLING_DOWN:
            progress = recovery_info.get('progress', 0.0)
            stability = recovery_info.get('stability', 0.0)
            reasons.append(f"해제 진행중 (진행도: {progress:.1%}, 안정성: {stability:.1%})")
        elif alert_level != AlertLevel.NORMAL and self.consecutive_normal_count > 0:
            reasons.append(f"정상 연속 {self.consecutive_normal_count}회")
        
        return " | ".join(reasons) if reasons else "정상 활동"
    
    def get_current_status(self) -> Dict[str, Any]:
        """현재 상태 반환 (알람 해제 정보 포함)"""
        current_time = time.time()
        
        # 회복 진행상황 계산
        recovery_progress = 0.0
        stability_score = 0.0
        if self.recovery_start_time > 0:
            recovery_time = current_time - self.recovery_start_time
            recovery_progress = min(1.0, recovery_time / self.min_recovery_time)
        
        stability_score = self._calculate_current_stability(current_time)
        
        return {
            'consecutive_count': self.consecutive_fight_count,
            'consecutive_normal_count': self.consecutive_normal_count,
            'recent_fight_ratio': self._calculate_recent_fight_ratio(),
            'adaptive_threshold': self.adaptive_threshold,
            'window_history_size': len(self.window_history),
            'time_since_last_fight': current_time - self.last_fight_time,
            'time_since_last_alert': current_time - self.last_alert_time,
            'total_fights': len(self.fight_timestamps),
            'total_alerts': len(self.alert_history),
            'is_cooldown': current_time - self.last_fight_time < self.cooldown_period,
            
            # 알람 해제 관련 정보
            'current_alert_level': self.current_alert_level.value,
            'alert_duration': current_time - self.alert_start_time if self.alert_start_time > 0 else 0,
            'recovery_progress': recovery_progress,
            'stability_score': stability_score,
            'is_recovering': self.recovery_start_time > 0,
            'recovery_duration': current_time - self.recovery_start_time if self.recovery_start_time > 0 else 0,
            'situation_stable': self._is_situation_stable(),
        }
    
    def _calculate_current_stability(self, current_time: float) -> float:
        """현재 안정성 점수 계산"""
        recent_ratio = self._calculate_recent_fight_ratio()
        
        stability_factors = {
            'normal_streak': min(1.0, self.consecutive_normal_count / self.normal_streak_required),
            'low_fight_ratio': max(0.0, 1.0 - (recent_ratio / self.fight_ratio_threshold)),
            'time_stability': min(1.0, (current_time - self.last_fight_time) / self.cooldown_period),
            'no_recent_fights': 1.0 if len([t for t in self.fight_timestamps 
                                          if current_time - t <= 60]) == 0 else 0.5
        }
        
        return sum(stability_factors.values()) / len(stability_factors)
    
    def reset_state(self):
        """상태 초기화 (알람 해제 상태 포함)"""
        self.window_history.clear()
        self.consecutive_fight_count = 0
        self.consecutive_normal_count = 0
        self.last_fight_time = 0
        self.last_alert_time = 0
        self.fight_timestamps.clear()
        self.alert_history.clear()
        self.adaptive_threshold = self.config.get('classification_threshold', 0.5)
        
        # 알람 해제 상태 초기화
        self.current_alert_level = AlertLevel.NORMAL
        self.alert_start_time = 0
        self.recovery_start_time = 0
        
        print("RealtimeDecisionEngine state reset (including alert recovery system)")


class RealtimeFightAnalyzer:
    """
    실시간 폭력 분석기 - 의사결정 엔진과 통계 분석 결합
    """
    
    def __init__(self, decision_config: Optional[Dict[str, Any]] = None):
        self.decision_engine = RealtimeDecisionEngine(decision_config)
        
        # 분석 통계
        self.analysis_stats = {
            'total_windows': 0,
            'fight_windows': 0,
            'alerts_by_level': {level.value: 0 for level in AlertLevel},
            'avg_confidence': 0.0,
            'max_consecutive': 0,
            'peak_fight_ratio': 0.0
        }
    
    def analyze_window(self, window_data: WindowData, 
                      prediction_result: Dict[str, Any]) -> Tuple[DecisionResult, Dict[str, Any]]:
        """윈도우 분석 및 통계 업데이트"""
        
        # 의사결정 실행
        decision = self.decision_engine.process_window(window_data, prediction_result)
        
        # 통계 업데이트
        self.analysis_stats['total_windows'] += 1
        
        if decision.is_fight:
            self.analysis_stats['fight_windows'] += 1
        
        self.analysis_stats['alerts_by_level'][decision.alert_level.value] += 1
        
        # 누적 평균 신뢰도
        total = self.analysis_stats['total_windows']
        current_avg = self.analysis_stats['avg_confidence']
        self.analysis_stats['avg_confidence'] = (
            (current_avg * (total - 1) + decision.confidence) / total
        )
        
        # 최대값 업데이트
        self.analysis_stats['max_consecutive'] = max(
            self.analysis_stats['max_consecutive'],
            decision.consecutive_count
        )
        
        self.analysis_stats['peak_fight_ratio'] = max(
            self.analysis_stats['peak_fight_ratio'],
            decision.recent_fight_ratio
        )
        
        # 현재 상태와 통계 반환
        current_status = self.decision_engine.get_current_status()
        analysis_result = {
            'decision': decision,
            'current_status': current_status,
            'analysis_stats': self.analysis_stats.copy()
        }
        
        return decision, analysis_result
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 반환"""
        stats = self.analysis_stats
        total = max(stats['total_windows'], 1)
        
        return {
            'total_windows_processed': stats['total_windows'],
            'overall_fight_ratio': stats['fight_windows'] / total,
            'average_confidence': stats['avg_confidence'],
            'alert_distribution': stats['alerts_by_level'],
            'max_consecutive_fights': stats['max_consecutive'],
            'peak_fight_ratio': stats['peak_fight_ratio'],
            'alert_rate': sum(stats['alerts_by_level'].values()) / total
        }


# 테스트 및 데모 코드
if __name__ == "__main__":
    import random
    
    # 테스트 설정
    test_config = {
        'consecutive_threshold': 3,
        'fight_ratio_threshold': 0.4,
        'sliding_window_size': 10,
        'cooldown_period': 20,
        'classification_threshold': 0.5
    }
    
    analyzer = RealtimeFightAnalyzer(test_config)
    
    print("Realtime Decision Engine Test")
    print("=" * 50)
    
    # 시뮬레이션 데이터로 테스트
    for i in range(20):
        # 가짜 윈도우 데이터
        from .realtime_window_manager import WindowData
        window_data = WindowData(
            window_idx=i,
            start_frame=i*50,
            end_frame=(i+1)*50,
            frames=[],
            annotation={}
        )
        
        # 가짜 예측 결과 (시간에 따라 변화)
        if 5 <= i <= 8:  # 연속 Fight 시뮬레이션
            fight_score = 0.8 + random.uniform(-0.1, 0.1)
            is_fight = True
        else:
            fight_score = 0.3 + random.uniform(-0.2, 0.2)
            is_fight = fight_score > 0.5
        
        prediction_result = {
            'fight_score': fight_score,
            'is_fight': is_fight,
            'confidence': abs(fight_score - 0.5) * 2  # 0.5에서 멀수록 신뢰도 높음
        }
        
        # 분석 실행
        decision, analysis = analyzer.analyze_window(window_data, prediction_result)
        
        print(f"Window {i:2d}: Fight={decision.is_fight}, "
              f"Level={decision.alert_level.value:10s}, "
              f"Conf={decision.confidence:.3f}, "
              f"Consec={decision.consecutive_count}, "
              f"Ratio={decision.recent_fight_ratio:.3f}")
        
        time.sleep(0.1)  # 시뮬레이션 딜레이
    
    print("\n" + "=" * 50)
    print("Final Performance Summary:")
    summary = analyzer.get_performance_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")