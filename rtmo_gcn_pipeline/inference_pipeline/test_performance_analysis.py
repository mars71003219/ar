#!/usr/bin/env python3
"""
Performance Analysis Test Script
성능 분석 테스트 스크립트 - 단계별 처리 시간 측정
"""

import os
import sys
import time
import logging
import numpy as np
from pathlib import Path

# 프로젝트 루트 경로 추가
sys.path.append('/workspace/rtmo_gcn_pipeline/inference_pipeline')

from config import INFERENCE_CONFIG, validate_config
from pose_estimator import RTMOPoseEstimator
from fight_tracker import FightPrioritizedTracker
from action_classifier import STGCNActionClassifier
from video_overlay import VideoOverlayGenerator
from performance_analyzer import PerformanceAnalyzer, StageTimer

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceTestPipeline:
    """성능 테스트용 파이프라인"""
    
    def __init__(self):
        # 성능 분석기 초기화
        self.analyzer = PerformanceAnalyzer()
        
        # 설정 검증
        validate_config()
        
        # 파이프라인 구성 요소 초기화
        self._init_components()
    
    def _init_components(self):
        """파이프라인 구성 요소 초기화"""
        logger.info("성능 테스트 파이프라인 구성 요소 초기화 중...")
        
        # 1. RTMO 포즈 추정기
        from config import POSE_CONFIG, POSE_CHECKPOINT, GCN_CONFIG, GCN_CHECKPOINT
        pose_config = POSE_CONFIG
        pose_checkpoint = POSE_CHECKPOINT
        
        with StageTimer(self.analyzer, "component_init_pose"):
            self.pose_estimator = RTMOPoseEstimator(pose_config, pose_checkpoint, 'cuda:0')
        
        # 2. Fight-우선 트래커 (ByteTrack 기반)
        bytetrack_config = {
            'high_thresh': 0.6,
            'low_thresh': 0.1,
            'max_disappeared': 30,
            'min_hits': 3
        }
        
        with StageTimer(self.analyzer, "component_init_tracker"):
            self.tracker = FightPrioritizedTracker(
                frame_width=INFERENCE_CONFIG['frame_extraction']['resize'][0],
                frame_height=INFERENCE_CONFIG['frame_extraction']['resize'][1],
                region_weights=INFERENCE_CONFIG['region_weights'],
                composite_weights=INFERENCE_CONFIG['composite_weights'],
                bytetrack_config=bytetrack_config
            )
        
        # 3. STGCN++ 분류기
        gcn_config = GCN_CONFIG
        gcn_checkpoint = GCN_CHECKPOINT
        
        with StageTimer(self.analyzer, "component_init_classifier"):
            self.classifier = STGCNActionClassifier(gcn_config, gcn_checkpoint, 'cuda:0')
        
        # 4. 비디오 오버레이 생성기 (선택적)
        with StageTimer(self.analyzer, "component_init_overlay"):
            self.overlay_generator = VideoOverlayGenerator(
                stgcn_joint_color=(0, 255, 128),
                other_joint_color=(255, 0, 0)
            )
            if hasattr(self.pose_estimator, 'model'):
                self.overlay_generator.init_visualizer_with_pose_model(self.pose_estimator.model)
        
        logger.info("성능 테스트 파이프라인 구성 요소 초기화 완료")
    
    def analyze_video_performance(self, video_path: str, generate_overlay: bool = True):
        """비디오 성능 분석"""
        logger.info(f"성능 분석 시작: {os.path.basename(video_path)}")
        
        total_start_time = time.time()
        
        try:
            # 1. RTMO 포즈 추정
            with StageTimer(self.analyzer, "pose_estimation"):
                pose_results = self.pose_estimator.estimate_poses_from_video(
                    video_path, 
                    max_frames=INFERENCE_CONFIG['frame_extraction']['max_frames']
                )
            
            if not pose_results:
                logger.error("포즈 추정 결과가 비어있음")
                return None
            
            total_frames = len(pose_results)
            logger.info(f"포즈 추정 완료: {total_frames}프레임")
            
            # 2. Fight-우선 트래킹 준비
            with StageTimer(self.analyzer, "tracking_preparation"):
                self.tracker.reset()
                
                detections_list = []
                keypoints_list = []
                scores_list = []
                
                for keypoints, scores in pose_results:
                    if keypoints is not None and scores is not None and len(keypoints) > 0:
                        keypoints_frame_list = [keypoints[i] for i in range(len(keypoints))]
                        scores_frame_list = [scores[i] for i in range(len(scores))]
                        
                        detections = self.tracker.create_detections_from_pose_results(
                            keypoints_frame_list, scores_frame_list
                        )
                        detections_list.append(detections)
                        keypoints_list.append(keypoints_frame_list)
                        scores_list.append(scores_frame_list)
                    else:
                        detections_list.append([])
                        keypoints_list.append([])
                        scores_list.append([])
            
            # 3. Fight-우선 트래킹 (ByteTrack 기반)
            with StageTimer(self.analyzer, "tracking_processing"):
                if detections_list and any(len(det) > 0 for det in detections_list):
                    selected_keypoints, selected_scores, track_ids_per_frame = \
                        self.tracker.process_video_sequence_with_detections(
                            detections_list, keypoints_list, scores_list,
                            sequence_length=INFERENCE_CONFIG['sequence_length'],
                            num_person=INFERENCE_CONFIG['num_person']
                        )
                else:
                    # 빈 비디오 처리
                    num_person = INFERENCE_CONFIG['num_person']
                    seq_len = INFERENCE_CONFIG['sequence_length'] or 30
                    selected_keypoints = np.zeros((seq_len, num_person, 17, 2))
                    selected_scores = np.zeros((seq_len, num_person, 17))
                    track_ids_per_frame = [[-1] * num_person for _ in range(seq_len)]
            
            logger.info(f"트래킹 완료: {selected_keypoints.shape}")
            
            # 4. STGCN++ 분류
            with StageTimer(self.analyzer, "stgcn_classification"):
                img_shape = INFERENCE_CONFIG['frame_extraction']['resize'][::-1]
                classification_result = self.classifier.classify_video_sequence(
                    selected_keypoints, selected_scores, 
                    window_size=INFERENCE_CONFIG['sequence_length'],
                    stride=INFERENCE_CONFIG['sequence_length'] // 2,
                    img_shape=img_shape
                )
            
            logger.info(f"분류 완료: {classification_result['prediction_label']} ({classification_result['confidence']:.3f})")
            
            # 5. 오버레이 비디오 생성 (선택적)
            overlay_path = None
            if generate_overlay:
                with StageTimer(self.analyzer, "overlay_generation"):
                    video_name = os.path.splitext(os.path.basename(video_path))[0]
                    overlay_dir = "/workspace/rtmo_gcn_pipeline/inference_pipeline/results/overlays"
                    os.makedirs(overlay_dir, exist_ok=True)
                    overlay_path = os.path.join(overlay_dir, f"{video_name}_performance_test_overlay.mp4")
                    
                    success = self.overlay_generator.create_overlay_with_bytetrack(
                        video_path=video_path,
                        pose_results=pose_results,
                        classification_result=classification_result,
                        output_path=overlay_path,
                        num_person=INFERENCE_CONFIG['num_person'],
                        tracker=self.tracker
                    )
                    
                    if not success:
                        overlay_path = None
                        logger.warning("오버레이 생성 실패")
            
            # 전체 처리 시간
            total_time = time.time() - total_start_time
            
            # 성능 분석 결과 생성
            analysis = self.analyzer.analyze_performance(total_frames)
            analysis['total_processing_time'] = total_time
            analysis['overall_fps'] = total_frames / total_time
            analysis['video_info'] = {
                'path': video_path,
                'frames': total_frames,
                'classification': classification_result,
                'overlay_path': overlay_path
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"성능 분석 실패: {e}")
            return None

def main():
    """메인 함수"""
    # 테스트 비디오 경로
    test_video = "/aivanas/raw/surveillance/action/violence/action_recognition/data/UCF_Crime_test/Fight/Fighting033_x264.mp4"
    
    if not os.path.exists(test_video):
        logger.error(f"테스트 비디오 파일이 존재하지 않음: {test_video}")
        return
    
    print("🚀 파이프라인 성능 분석 시작")
    print(f"📹 테스트 비디오: {os.path.basename(test_video)}")
    
    # 성능 테스트 파이프라인 생성
    pipeline = PerformanceTestPipeline()
    
    # 성능 분석 실행
    analysis = pipeline.analyze_video_performance(test_video, generate_overlay=True)
    
    if analysis:
        # 결과 출력
        pipeline.analyzer.print_summary(analysis)
        
        # 결과 저장
        output_path = "/workspace/rtmo_gcn_pipeline/inference_pipeline/results/performance_analysis.json"
        pipeline.analyzer.save_analysis(analysis, output_path)
        
        print(f"\n💾 상세 분석 결과 저장: {output_path}")
    else:
        print("❌ 성능 분석 실패")

if __name__ == "__main__":
    main()