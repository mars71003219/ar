#!/usr/bin/env python3
"""
Overlay Renderer - UI 오버레이 렌더링 모듈
"""

import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

try:
    from configs.visualizer_config import config as default_config
except ImportError:
    try:
        from ..configs.visualizer_config import config as default_config
    except ImportError:
        print("Warning: Could not import visualizer config for overlay renderer. Using defaults.")
        default_config = None


class OverlayRenderer:
    """
UI 오버레이 렌더링 클래스
윈도우 감지 결과, 트랙 정보, 통계 등을 화면에 표시
"""
    
    def __init__(self, config=None):
        # 설정 로드
        self.config = config or default_config
        
        # 폰트 설정
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        if self.config:
            # 설정 파일에서 값 로드
            self.font_scale = self.config.font_scale
            self.font_thickness = self.config.font_thickness
            self.text_color = self.config.text_color
            self.bg_color = self.config.bg_color
            self.fight_color = self.config.fight_color
            self.nonfight_color = self.config.nonfight_color
            self.box_padding = self.config.box_padding
        else:
            # 폴백 설정
            self.font_scale = 0.7
            self.font_thickness = 2
            self.text_color = (255, 255, 255)
            self.bg_color = (0, 0, 0)
            self.fight_color = (0, 0, 255)
            self.nonfight_color = (0, 255, 0)
            self.box_padding = 10
    
    def draw_text_with_background(self, image: np.ndarray, text: str, 
                                 position: Tuple[int, int], 
                                 text_color: Optional[Tuple[int, int, int]] = None,
                                 bg_color: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
        """
        배경이 있는 텍스트 그리기
        
        Args:
            image: 대상 이미지
            text: 표시할 텍스트
            position: 텍스트 위치 (x, y)
            text_color: 텍스트 색상
            bg_color: 배경 색상
        
        Returns:
            텍스트가 그려진 이미지
        """
        if text_color is None:
            text_color = self.text_color
        if bg_color is None:
            bg_color = self.bg_color[:3]  # 알파 채널 제거
        
        # 텍스트 크기 계산
        text_size = cv2.getTextSize(text, self.font, self.font_scale, self.font_thickness)[0]
        
        # 배경 사각형 그리기
        x, y = position
        cv2.rectangle(image, 
                     (x - self.box_padding, y - text_size[1] - self.box_padding),
                     (x + text_size[0] + self.box_padding, y + self.box_padding),
                     bg_color, -1)
        
        # 텍스트 그리기
        cv2.putText(image, text, (x, y), self.font, self.font_scale, 
                   text_color, self.font_thickness)
        
        return image
    
    def draw_window_detection_info(self, image: np.ndarray, 
                                  window_result: Dict[str, Any],
                                  prediction_result: Optional[str] = None) -> np.ndarray:
        """
        윈도우 감지 정보 표시
        
        Args:
            image: 대상 이미지
            window_result: 윈도우 결과 데이터
            prediction_result: 예측 결과 (옵션)
        
        Returns:
            정보가 표시된 이미지
        """
        result_image = image.copy()
        
        try:
            # 윈도우 기본 정보
            if 'window_idx' in window_result:
                window_info = f"Window: {window_result['window_idx']}"
                result_image = self.draw_text_with_background(
                    result_image, window_info, (10, 30))
            
            if 'start_frame' in window_result and 'end_frame' in window_result:
                frame_info = f"Frame: {window_result['start_frame']}-{window_result['end_frame']}"
                result_image = self.draw_text_with_background(
                    result_image, frame_info, (10, 60))
            
            # 예측 결과 표시
            if prediction_result:
                pred_color = self.fight_color if prediction_result.lower() == 'fight' else self.nonfight_color
                result_image = self.draw_text_with_background(
                    result_image, f"Prediction: {prediction_result}", 
                    (10, 90), text_color=pred_color)
            
            # 사람 수 정보
            if ('annotation' in window_result and 
                'persons' in window_result['annotation']):
                person_count = len(window_result['annotation']['persons'])
                result_image = self.draw_text_with_background(
                    result_image, f"Persons: {person_count}", (10, 120))
            
        except Exception as e:
            print(f"Error drawing window detection info: {e}")
        
        return result_image
    
    def draw_track_info(self, image: np.ndarray, track_id: int, 
                       bbox: List[float], score: float = None,
                       position_offset: Tuple[int, int] = (0, 0)) -> np.ndarray:
        """
        트랙 정보 표시
        
        Args:
            image: 대상 이미지
            track_id: 트랙 ID
            bbox: 바운딩 박스 [x1, y1, x2, y2]
            score: 신룰도 점수 (옵션)
            position_offset: 위치 오프셋
        
        Returns:
            트랙 정보가 표시된 이미지
        """
        result_image = image.copy()
        
        try:
            if len(bbox) >= 4:
                x1, y1, x2, y2 = map(int, bbox[:4])
                
                # 바운딩 박스 그리기
                color = self._get_track_color(track_id)
                cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
                
                # 트랙 ID 표시
                track_text = f"ID: {track_id}"
                if score is not None:
                    track_text += f" ({score:.2f})"
                
                text_pos = (x1 + position_offset[0], y1 - 10 + position_offset[1])
                result_image = self.draw_text_with_background(
                    result_image, track_text, text_pos, text_color=color)
                
        except Exception as e:
            print(f"Error drawing track info: {e}")
        
        return result_image
    
    def draw_statistics(self, image: np.ndarray, stats: Dict[str, Any],
                       position: Tuple[int, int] = None) -> np.ndarray:
        """
        통계 정보 표시
        
        Args:
            image: 대상 이미지
            stats: 통계 데이터
            position: 시작 위치 (옵션)
        
        Returns:
            통계 정보가 표시된 이미지
        """
        result_image = image.copy()
        
        if position is None:
            position = (image.shape[1] - 200, 30)  # 오른쪽 상단
        
        try:
            y_offset = 0
            for key, value in stats.items():
                stat_text = f"{key}: {value}"
                text_pos = (position[0], position[1] + y_offset)
                result_image = self.draw_text_with_background(
                    result_image, stat_text, text_pos)
                y_offset += 30
                
        except Exception as e:
            print(f"Error drawing statistics: {e}")
        
        return result_image
    
    def draw_classification_result(self, image: np.ndarray, 
                                  result: str, confidence: float = None,
                                  position: Tuple[int, int] = None) -> np.ndarray:
        """
        분류 결과 표시
        
        Args:
            image: 대상 이미지
            result: 분류 결과 (Fight/NonFight)
            confidence: 신룰도 (옵션)
            position: 표시 위치 (옵션)
        
        Returns:
            분류 결과가 표시된 이미지
        """
        result_image = image.copy()
        
        if position is None:
            position = (10, image.shape[0] - 30)  # 왼쪽 하단
        
        try:
            # 결과에 따른 색상 선택
            color = self.fight_color if result.lower() == 'fight' else self.nonfight_color
            
            # 텍스트 구성
            text = f"Result: {result}"
            if confidence is not None:
                text += f" ({confidence:.2f})"
            
            result_image = self.draw_text_with_background(
                result_image, text, position, text_color=color)
                
        except Exception as e:
            print(f"Error drawing classification result: {e}")
        
        return result_image
    
    def _get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """트랙 ID에 따른 색상 반환"""
        if self.config and hasattr(self.config, 'get_color'):
            return self.config.get_color(track_id)
        else:
            # 폴백 색상
            colors = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
                (0, 128, 128), (128, 128, 0), (75, 0, 130), (220, 20, 60)
            ]
            return colors[track_id % len(colors)]
