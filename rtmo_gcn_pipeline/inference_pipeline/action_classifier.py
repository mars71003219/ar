#!/usr/bin/env python3
"""
STGCN++ Action Classification Module
STGCN++ 행동 분류 모듈 - 학습된 모델을 사용한 폭력 행동 분류
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
import logging

from mmaction.apis import init_recognizer, inference_skeleton

logger = logging.getLogger(__name__)

class STGCNActionClassifier:
    """
    STGCN++ 행동 분류기
    학습된 STGCN++ 모델을 사용하여 Fight/NonFight 분류
    """
    
    def __init__(self, config_path: str, checkpoint_path: str, device: str = 'cuda:0'):
        """
        초기화
        
        Args:
            config_path: STGCN++ 모델 설정 파일 경로
            checkpoint_path: 학습된 모델 체크포인트 경로
            device: 추론 디바이스
        """
        self.device = device
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        
        # 클래스 매핑
        self.class_mapping = {
            0: 'NonFight',
            1: 'Fight'
        }
        
        # 모델 로드
        logger.info("STGCN++ 모델 로딩 중...")
        self.model = self._load_model()
        logger.info("STGCN++ 모델 로딩 완료")
        
    def _load_model(self):
        """STGCN++ 모델 로드"""
        try:
            model = init_recognizer(
                self.config_path,
                self.checkpoint_path,
                device=self.device
            )
            return model
        except Exception as e:
            logger.error(f"STGCN++ 모델 로드 실패: {e}")
            raise
    
    def preprocess_for_stgcn(self, keypoints: np.ndarray, scores: np.ndarray, 
                           confidence_threshold: float = 0.3) -> List[Dict]:
        """
        STGCN++ 입력 형식으로 전처리
        
        Args:
            keypoints: 키포인트 좌표 (T, num_person, 17, 2) 또는 (T, 17, 2)
            scores: 키포인트 신뢰도 (T, num_person, 17) 또는 (T, 17)
            confidence_threshold: 신뢰도 임계값
            
        Returns:
            MMAction2 형식의 포즈 결과 리스트
        """
        pose_results = []
        
        # 단일 인물 데이터인 경우 차원 확장
        if len(keypoints.shape) == 3:  # (T, 17, 2) -> (T, 1, 17, 2)
            keypoints = keypoints[:, np.newaxis, :, :]
            scores = scores[:, np.newaxis, :]
        
        for i in range(keypoints.shape[0]):
            frame_keypoints = keypoints[i]  # (num_person, 17, 2)
            frame_scores = scores[i]        # (num_person, 17)
            
            # 낮은 신뢰도 키포인트 필터링
            mask = frame_scores > confidence_threshold
            filtered_keypoints = frame_keypoints.copy()
            filtered_keypoints[~mask] = 0
            
            frame_result = {
                'keypoints': filtered_keypoints,
                'keypoint_scores': frame_scores
            }
            pose_results.append(frame_result)
        
        return pose_results
    
    def classify_sequence(self, keypoints: np.ndarray, scores: np.ndarray, 
                         img_shape: Tuple[int, int] = (480, 640)) -> Tuple[int, float, np.ndarray]:
        """
        키포인트 시퀀스를 분류
        
        Args:
            keypoints: 키포인트 좌표 (T, num_person, 17, 2) 또는 (T, 17, 2)
            scores: 키포인트 신뢰도 (T, num_person, 17) 또는 (T, 17)
            img_shape: 이미지 크기 (height, width)
            
        Returns:
            (prediction, confidence, score_distribution)
            - prediction: 예측 클래스 (0 또는 1)
            - confidence: 예측 신뢰도
            - score_distribution: 클래스별 점수 분포
        """
        try:
            # STGCN++ 입력 형식으로 전처리
            pose_results = self.preprocess_for_stgcn(keypoints, scores)
            
            # STGCN++ 추론
            result = inference_skeleton(
                model=self.model,
                pose_results=pose_results,
                img_shape=img_shape
            )
            
            # 결과 추출
            if hasattr(result, 'pred_score'):
                pred_scores = result.pred_score.cpu().numpy()
                confidence = float(np.max(pred_scores))
                prediction = int(np.argmax(pred_scores))
                
                return prediction, confidence, pred_scores
            else:
                logger.warning("STGCN++ 결과에서 pred_score를 찾을 수 없습니다")
                return 0, 0.5, np.array([0.5, 0.5])
                
        except Exception as e:
            logger.error(f"STGCN++ 분류 실패: {e}")
            return 0, 0.5, np.array([0.5, 0.5])
    
    def classify_video_sequence(self, keypoints: np.ndarray, scores: np.ndarray,
                              window_size: int = 30, stride: int = 15,
                              img_shape: Tuple[int, int] = (480, 640)) -> Dict:
        """
        비디오 시퀀스에서 윈도우 기반 분류 (overlapping windows)
        
        Args:
            keypoints: 키포인트 좌표 (T, num_person, 17, 2) 또는 (T, 17, 2)
            scores: 키포인트 신뢰도 (T, num_person, 17) 또는 (T, 17)
            window_size: 윈도우 크기
            stride: 윈도우 간격
            img_shape: 이미지 크기
            
        Returns:
            분류 결과 딕셔너리
        """
        # 단일 인물 데이터인 경우 차원 확장
        if len(keypoints.shape) == 3:  # (T, 17, 2) -> (T, 1, 17, 2)
            keypoints = keypoints[:, np.newaxis, :, :]
            scores = scores[:, np.newaxis, :]
        
        if len(keypoints) < window_size:
            # 시퀀스가 윈도우보다 짧으면 패딩
            needed = window_size - len(keypoints)
            if len(keypoints) > 0:
                last_keypoints = keypoints[-1:]
                last_scores = scores[-1:]
                for _ in range(needed):
                    keypoints = np.concatenate([keypoints, last_keypoints], axis=0)
                    scores = np.concatenate([scores, last_scores], axis=0)
            else:
                # 완전히 빈 시퀀스 (num_person 차원 고려)
                num_person = 1  # 기본값
                keypoints = np.zeros((window_size, num_person, 17, 2))
                scores = np.zeros((window_size, num_person, 17))
        
        window_results = []
        
        # 윈도우 기반 추론
        for start_idx in range(0, len(keypoints) - window_size + 1, stride):
            end_idx = start_idx + window_size
            
            window_keypoints = keypoints[start_idx:end_idx]
            window_scores_data = scores[start_idx:end_idx]
            
            prediction, confidence, score_dist = self.classify_sequence(
                window_keypoints, window_scores_data, img_shape
            )
            
            # 간소화된 구조: prediction과 scores를 하나로 통합
            if hasattr(score_dist, 'tolist'):
                scores_list = score_dist.tolist()
            else:
                scores_list = score_dist
            
            window_results.append({
                "pred": prediction,
                "scores": scores_list
            })
        
        # 최종 결과 결정 (majority voting + confidence weighting)
        if window_results:
            # 간소화된 구조에서 데이터 추출
            window_predictions = [w["pred"] for w in window_results]
            window_confidences = [max(w["scores"]) for w in window_results]
            
            # 신뢰도 가중 투표
            weighted_votes = sum(pred * conf for pred, conf in zip(window_predictions, window_confidences))
            total_confidence = sum(window_confidences)
            
            if total_confidence > 0:
                final_score = weighted_votes / total_confidence
                final_prediction = 1 if final_score > 0.5 else 0
                final_confidence = total_confidence / len(window_confidences)
            else:
                final_prediction = 0
                final_confidence = 0.5
                final_score = 0.5
        else:
            final_prediction = 0
            final_confidence = 0.5
            final_score = 0.5
            window_predictions = []
        
        # 연속 윈도우 기반 최종 예측 (사용자 설정 가능)
        try:
            from .config import INFERENCE_CONFIG
        except ImportError:
            from config import INFERENCE_CONFIG
        consecutive_fight_threshold = INFERENCE_CONFIG.get('consecutive_fight_threshold', 5)
        consecutive_final_prediction = self._apply_consecutive_window_logic(
            window_predictions, consecutive_fight_threshold
        )
        
        return {
            'prediction': final_prediction,
            'confidence': final_confidence,
            'prediction_label': self.class_mapping[final_prediction],
            'final_score': final_score,
            'window_results': window_results,  # 최적화된 구조만 사용
            'num_windows': len(window_results),
            'consecutive_prediction': consecutive_final_prediction,
            'consecutive_prediction_label': self.class_mapping[consecutive_final_prediction],
            'consecutive_threshold': consecutive_fight_threshold
        }
    
    def _apply_consecutive_window_logic(self, window_predictions: List[int], 
                                      consecutive_threshold: int = 5) -> int:
        """
        연속 윈도우 기반 최종 예측
        
        Args:
            window_predictions: 윈도우별 예측 결과 리스트
            consecutive_threshold: 연속 Fight 예측 임계값
            
        Returns:
            최종 예측 결과 (0: NonFight, 1: Fight)
        """
        if not window_predictions:
            return 0
            
        # 연속된 Fight 예측 구간 찾기
        max_consecutive_fight = 0
        current_consecutive_fight = 0
        
        for prediction in window_predictions:
            if prediction == 1:  # Fight
                current_consecutive_fight += 1
                max_consecutive_fight = max(max_consecutive_fight, current_consecutive_fight)
            else:  # NonFight
                current_consecutive_fight = 0
        
        # 임계값 이상의 연속 Fight가 있으면 Fight로 판정
        return 1 if max_consecutive_fight >= consecutive_threshold else 0
    
    def batch_classify(self, keypoints_batch: List[np.ndarray], scores_batch: List[np.ndarray],
                      img_shape: Tuple[int, int] = (480, 640)) -> List[Dict]:
        """
        배치 분류
        
        Args:
            keypoints_batch: 키포인트 배치 리스트
            scores_batch: 점수 배치 리스트
            img_shape: 이미지 크기
            
        Returns:
            분류 결과 리스트
        """
        results = []
        
        for keypoints, scores in zip(keypoints_batch, scores_batch):
            result = self.classify_video_sequence(keypoints, scores, img_shape=img_shape)
            results.append(result)
        
        return results
    
    def get_class_name(self, class_idx: int) -> str:
        """클래스 인덱스를 클래스 이름으로 변환"""
        return self.class_mapping.get(class_idx, 'Unknown')
    
    def analyze_prediction_confidence(self, result: Dict) -> Dict:
        """
        예측 신뢰도 분석
        
        Args:
            result: classify_video_sequence 결과
            
        Returns:
            신뢰도 분석 결과
        """
        # window_results에서 신뢰도와 예측 추출
        window_results = result.get('window_results', [])
        
        if not window_results:
            return {
                'confidence_mean': 0.0,
                'confidence_std': 0.0,
                'confidence_min': 0.0,
                'confidence_max': 0.0,
                'prediction_consistency': 0.0
            }
        
        # window_results에서 신뢰도와 예측 데이터 추출
        window_confidences = [max(w["scores"]) for w in window_results]
        window_predictions = [w["pred"] for w in window_results]
        
        confidence_mean = np.mean(window_confidences)
        confidence_std = np.std(window_confidences)
        confidence_min = np.min(window_confidences)
        confidence_max = np.max(window_confidences)
        
        # 예측 일관성 (같은 예측을 한 윈도우의 비율)
        most_common_pred = max(set(window_predictions), key=window_predictions.count)
        consistency = window_predictions.count(most_common_pred) / len(window_predictions)
        
        return {
            'confidence_mean': float(confidence_mean),
            'confidence_std': float(confidence_std),
            'confidence_min': float(confidence_min),
            'confidence_max': float(confidence_max),
            'prediction_consistency': float(consistency)
        }