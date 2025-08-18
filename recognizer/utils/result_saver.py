"""
결과 저장 유틸리티
분석 결과를 JSON, PKL 형식으로 저장하는 공용 모듈
"""

import json
import pickle
import time
import logging
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class ResultSaver:
    """분석 결과 저장 공용 클래스"""
    
    @staticmethod
    def save_analysis_results(input_file: str, output_dir: str, result_dict: Dict[str, Any]) -> bool:
        """분석 결과를 JSON, PKL 파일로 저장"""
        try:
            video_path = Path(input_file)
            video_name = video_path.stem
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 윈도우 프레임 범위 정보 추가
            classifications_with_frames = []
            for i, classification in enumerate(result_dict.get('classifications', [])):
                # 윈도우 프레임 범위 계산
                window_start = i * 50  # stride=50 기준
                window_end = window_start + 100 - 1  # window_size=100 기준
                
                # 분류 결과에 프레임 범위 추가
                enhanced_classification = classification.copy()
                enhanced_classification.update({
                    'window_id': i,
                    'window_start_frame': window_start,
                    'window_end_frame': window_end,
                    'frame_range': f"{window_start}-{window_end}"
                })
                classifications_with_frames.append(enhanced_classification)
            
            # JSON 결과 생성
            json_result = {
                'input_video': input_file,
                'video_name': video_name,
                'performance_stats': result_dict.get('performance_stats', {}),
                'classification_results': classifications_with_frames,
                'window_analysis': {
                    'total_windows': len(classifications_with_frames),
                    'window_size': 100,
                    'window_stride': 50,
                    'windows_info': [
                        {
                            'window_id': i,
                            'frame_range': f"{i * 50}-{i * 50 + 99}",
                            'prediction': cls.get('predicted_label', 'Unknown'),
                            'confidence': cls.get('confidence', 0.0)
                        } for i, cls in enumerate(classifications_with_frames)
                    ]
                },
                'summary': {
                    'total_classifications': len(classifications_with_frames),
                    'fight_predictions': sum(1 for c in classifications_with_frames if c.get('prediction') == 1),
                    'non_fight_predictions': sum(1 for c in classifications_with_frames if c.get('prediction') == 0),
                    'total_pose_frames': result_dict.get('total_frames', 0)
                },
                'timestamp': time.time()
            }
            
            # JSON 파일 저장
            json_file = output_path / f"{video_name}_results.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(json_result, f, indent=2, ensure_ascii=False)
            
            # PKL 파일들 저장
            if 'raw_pose_results' in result_dict:
                # 1. 포즈추정 PKL 파일 (원본 포즈 데이터)
                rtmo_poses_file = output_path / f"{video_name}_rtmo_poses.pkl"
                with open(rtmo_poses_file, 'wb') as f:
                    pickle.dump(result_dict['raw_pose_results'], f)
                logger.info(f"Raw pose results saved to: {rtmo_poses_file}")
            
            if 'processed_frame_poses' in result_dict:
                # 2. 트래킹+복합점수 계산 후 정렬된 PKL 파일
                frame_poses_file = output_path / f"{video_name}_frame_poses.pkl"
                with open(frame_poses_file, 'wb') as f:
                    pickle.dump(result_dict['processed_frame_poses'], f)
                logger.info(f"Processed frame poses saved to: {frame_poses_file}")
            
            logger.info(f"Analysis results saved to: {json_file}")
            logger.info(f"Total windows classified: {len(classifications_with_frames)}")
            logger.info(f"Total frames processed: {result_dict.get('total_frames', 0)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save analysis results: {e}")
            return False
    
    @staticmethod
    def create_summary_stats(classifications: List[Dict[str, Any]]) -> Dict[str, Any]:
        """분류 결과 요약 통계 생성"""
        if not classifications:
            return {
                'total_classifications': 0,
                'fight_predictions': 0,
                'non_fight_predictions': 0,
                'average_confidence': 0.0,
                'max_confidence': 0.0,
                'min_confidence': 0.0
            }
        
        fight_count = sum(1 for c in classifications if c.get('prediction') == 1)
        non_fight_count = len(classifications) - fight_count
        confidences = [c.get('confidence', 0.0) for c in classifications]
        
        return {
            'total_classifications': len(classifications),
            'fight_predictions': fight_count,
            'non_fight_predictions': non_fight_count,
            'fight_ratio': fight_count / len(classifications) if classifications else 0,
            'average_confidence': sum(confidences) / len(confidences) if confidences else 0,
            'max_confidence': max(confidences) if confidences else 0,
            'min_confidence': min(confidences) if confidences else 0
        }