#!/usr/bin/env python3
"""
Enhanced ByteTracker with RTMO - Working Demo
import 문제를 완전히 해결한 데모
"""

import os
import sys
import cv2
import numpy as np
import time
import colorsys
from pathlib import Path
from enum import Enum
from typing import List, Tuple, Optional, Dict
from collections import deque
import argparse

# 경로 설정
sys.path.insert(0, '/workspace')
tracker_root = Path(__file__).parent

try:
    from mmpose.apis import init_model, inference_bottomup
    import torch
    from scipy.optimize import linear_sum_assignment
    print("✅ All dependencies imported successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


# === RTMO 설정 ===
RTMO_CONFIG = '/workspace/mmpose/configs/body_2d_keypoint/rtmo/body7/rtmo-m_16xb16-600e_body7-640x640.py'
RTMO_CHECKPOINT = '/workspace/mmpose/checkpoints/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.pth'

# === 설정 클래스들 ===
class DefaultTrackerConfig:
    """기본 트래커 설정"""
    obj_score_thrs = {'high': 0.6, 'low': 0.1}
    init_track_thr = 0.7
    weight_iou_with_det_scores = True
    match_iou_thrs = {'high': 0.1, 'low': 0.5, 'tentative': 0.3}
    num_tentatives = 3
    num_frames_retain = 30
    
    @classmethod
    def get_config_dict(cls):
        return {
            'obj_score_thrs': cls.obj_score_thrs,
            'init_track_thr': cls.init_track_thr,
            'weight_iou_with_det_scores': cls.weight_iou_with_det_scores,
            'match_iou_thrs': cls.match_iou_thrs,
            'num_tentatives': cls.num_tentatives,
            'num_frames_retain': cls.num_frames_retain
        }


class RTMOTrackerConfig(DefaultTrackerConfig):
    """RTMO 최적화 설정"""
    obj_score_thrs = {'high': 0.5, 'low': 0.1}
    init_track_thr = 0.6
    match_iou_thrs = {'high': 0.1, 'low': 0.4, 'tentative': 0.3}
    num_tentatives = 2
    num_frames_retain = 50


# === 트래킹 클래스들 ===
class TrackState(Enum):
    NEW = 1
    TRACKED = 2
    LOST = 3
    REMOVED = 4


class SimpleKalmanFilter:
    """간단한 칼만 필터"""
    def __init__(self):
        self.state = np.zeros(8, dtype=np.float32)  # [cx, cy, w, h, vcx, vcy, vw, vh]
        self.initialized = False
        
    def initiate(self, bbox):
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        self.state = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32)
        self.initialized = True
        
    def predict(self):
        if not self.initialized:
            return
        dt = 1.0
        self.state[0] += self.state[4] * dt
        self.state[1] += self.state[5] * dt
        self.state[2] += self.state[6] * dt
        self.state[3] += self.state[7] * dt
        
    def update(self, bbox):
        if not self.initialized:
            self.initiate(bbox)
            return
            
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        
        alpha = 0.7
        self.state[4] = alpha * (cx - self.state[0]) + (1 - alpha) * self.state[4]
        self.state[5] = alpha * (cy - self.state[1]) + (1 - alpha) * self.state[5]
        self.state[6] = alpha * (w - self.state[2]) + (1 - alpha) * self.state[6]
        self.state[7] = alpha * (h - self.state[3]) + (1 - alpha) * self.state[7]
        
        self.state[0] = cx
        self.state[1] = cy
        self.state[2] = w
        self.state[3] = h
        
    def get_current_bbox(self):
        if not self.initialized:
            return np.array([0, 0, 0, 0])
        cx, cy, w, h = self.state[:4]
        return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])


class Track:
    """트랙 클래스"""
    count = 0
    
    def __init__(self, bbox, score):
        Track.count += 1
        self.track_id = Track.count
        self.kalman = SimpleKalmanFilter()
        self.kalman.initiate(bbox)
        self.score = score
        self.state = TrackState.NEW
        self.age = 0
        self.hits = 1
        self.hit_streak = 1
        self.time_since_update = 0
        
    def predict(self):
        self.kalman.predict()
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        
    def update(self, bbox, score):
        self.kalman.update(bbox)
        self.score = score
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        
        if self.state == TrackState.NEW and self.hits >= 3:
            self.state = TrackState.TRACKED
        elif self.state == TrackState.LOST:
            self.state = TrackState.TRACKED
            
    def mark_missed(self):
        if self.state == TrackState.TRACKED:
            self.state = TrackState.LOST
            
    def get_bbox(self):
        return self.kalman.get_current_bbox()


class EnhancedByteTracker:
    """향상된 ByteTracker"""
    def __init__(self, config):
        self.config = config
        self.obj_score_thrs = config['obj_score_thrs']
        self.match_iou_thrs = config['match_iou_thrs']
        self.num_frames_retain = config['num_frames_retain']
        
        self.tracked_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []
        self.frame_count = 0
        
        # 통계
        self.stats = {'total_tracks': 0, 'total_detections': 0}
        
    def compute_iou(self, bbox1, bbox2):
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def associate_detections_to_trackers(self, detections, trackers, iou_threshold):
        if len(trackers) == 0:
            return [], list(range(len(detections))), []
        if len(detections) == 0:
            return [], [], list(range(len(trackers)))
            
        iou_matrix = np.zeros((len(detections), len(trackers)))
        for d, det in enumerate(detections):
            for t, tracker in enumerate(trackers):
                iou_matrix[d, t] = self.compute_iou(det[:4], tracker.get_bbox())
        
        cost_matrix = 1 - iou_matrix
        
        if min(cost_matrix.shape) > 0:
            det_indices, trk_indices = linear_sum_assignment(cost_matrix)
            matches = []
            for d, t in zip(det_indices, trk_indices):
                if iou_matrix[d, t] >= iou_threshold:
                    matches.append([d, t])
            
            unmatched_dets = [d for d in range(len(detections)) 
                             if d not in det_indices or 
                             (d in det_indices and 
                              iou_matrix[d, trk_indices[list(det_indices).index(d)]] < iou_threshold)]
            unmatched_trks = [t for t in range(len(trackers)) 
                             if t not in trk_indices or
                             (t in trk_indices and 
                              iou_matrix[det_indices[list(trk_indices).index(t)], t] < iou_threshold)]
        else:
            matches = []
            unmatched_dets = list(range(len(detections)))
            unmatched_trks = list(range(len(trackers)))
        
        return matches, unmatched_dets, unmatched_trks
    
    def update(self, detections, image_shape=None):
        """메인 업데이트 메서드"""
        self.frame_count += 1
        self.stats['total_detections'] += len(detections)
        
        # 예측 스텝
        for track in self.tracked_tracks + self.lost_tracks:
            track.predict()
        
        if len(detections) == 0:
            # detection이 없으면 기존 트랙들만 업데이트
            for track in self.tracked_tracks[:]:
                track.mark_missed()
                if track.time_since_update > self.num_frames_retain:
                    self.tracked_tracks.remove(track)
                    self.removed_tracks.append(track)
            return []
        
        # 높은/낮은 점수 detection 분리
        high_dets = detections[detections[:, 4] >= self.obj_score_thrs['high']]
        low_dets = detections[(detections[:, 4] >= self.obj_score_thrs['low']) & 
                             (detections[:, 4] < self.obj_score_thrs['high'])]
        
        # 첫 번째 매칭: 높은 점수 detections와 tracked tracks
        matches, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(
            high_dets, self.tracked_tracks, self.match_iou_thrs['high'])
        
        # 매칭된 트랙 업데이트
        for match in matches:
            det_idx, trk_idx = match
            self.tracked_tracks[trk_idx].update(high_dets[det_idx][:4], high_dets[det_idx][4])
        
        # 매칭되지 않은 tracked tracks를 lost로 이동
        for trk_idx in sorted(unmatched_trks, reverse=True):
            track = self.tracked_tracks[trk_idx]
            track.mark_missed()
            self.lost_tracks.append(track)
            del self.tracked_tracks[trk_idx]
        
        # 두 번째 매칭: 낮은 점수 detections와 lost tracks  
        if len(low_dets) > 0 and len(self.lost_tracks) > 0:
            matches2, unmatched_dets2, unmatched_trks2 = self.associate_detections_to_trackers(
                low_dets, self.lost_tracks, self.match_iou_thrs['low'])
            
            # 재매칭된 lost tracks를 tracked로 복귀
            for match in sorted(matches2, key=lambda x: x[1], reverse=True):
                det_idx, trk_idx = match
                track = self.lost_tracks[trk_idx]
                track.update(low_dets[det_idx][:4], low_dets[det_idx][4])
                self.tracked_tracks.append(track)
                del self.lost_tracks[trk_idx]
        
        # 새로운 트랙 생성 (매칭되지 않은 높은 점수 detections)
        for det_idx in unmatched_dets:
            if det_idx < len(high_dets):
                new_track = Track(high_dets[det_idx][:4], high_dets[det_idx][4])
                self.tracked_tracks.append(new_track)
                self.stats['total_tracks'] += 1
        
        # 오래된 lost tracks 제거
        for track in self.lost_tracks[:]:
            if track.time_since_update > self.num_frames_retain:
                self.lost_tracks.remove(track)
                self.removed_tracks.append(track)
        
        # TRACKED 상태인 트랙들만 반환
        active_tracks = [t for t in self.tracked_tracks if t.state == TrackState.TRACKED]
        return active_tracks
    
    def get_stats(self):
        return self.stats


# === 파이프라인 클래스들 ===
class RTMOTrackingPipeline:
    """RTMO + ByteTracker 통합 파이프라인"""
    def __init__(self, rtmo_config, rtmo_checkpoint, device='cuda:0', tracker_config=None):
        self.device = device
        
        # RTMO 모델 초기화
        print(f"Loading RTMO model from {rtmo_checkpoint}")
        self.rtmo_model = init_model(rtmo_config, rtmo_checkpoint, device=device)
        
        # ByteTracker 초기화
        if tracker_config is None:
            tracker_config = RTMOTrackerConfig.get_config_dict()
        
        self.tracker = EnhancedByteTracker(tracker_config)
        self.frame_count = 0
        self.stats = {'total_frames': 0, 'total_detections': 0}
        
    def extract_detections_from_pose_result(self, pose_result):
        """포즈 결과에서 detection 추출"""
        if not pose_result or not hasattr(pose_result, 'pred_instances'):
            return np.empty((0, 5))
        
        pred_instances = pose_result.pred_instances
        
        if not hasattr(pred_instances, 'bboxes') or len(pred_instances.bboxes) == 0:
            return np.empty((0, 5))
        
        # 바운딩 박스 추출
        if hasattr(pred_instances.bboxes, 'cpu'):
            bboxes = pred_instances.bboxes.cpu().numpy()
        else:
            bboxes = np.array(pred_instances.bboxes)
        
        # 점수 추출
        if hasattr(pred_instances, 'bbox_scores'):
            if hasattr(pred_instances.bbox_scores, 'cpu'):
                scores = pred_instances.bbox_scores.cpu().numpy()
            else:
                scores = np.array(pred_instances.bbox_scores)
        else:
            scores = np.ones(len(bboxes))
        
        return np.column_stack([bboxes, scores])
    
    def process_frame(self, image):
        """단일 프레임 처리"""
        self.frame_count += 1
        
        # RTMO로 포즈 추정
        pose_results = inference_bottomup(self.rtmo_model, image)
        
        if not pose_results:
            detections = np.empty((0, 5))
        else:
            detections = self.extract_detections_from_pose_result(pose_results[0])
        
        # ByteTracker로 추적
        tracks = self.tracker.update(detections, image.shape)
        
        # 통계 업데이트
        self.stats['total_frames'] += 1
        self.stats['total_detections'] += len(detections)
        
        return tracks, pose_results[0] if pose_results else None
    
    def get_stats(self):
        stats = self.stats.copy()
        stats.update(self.tracker.get_stats())
        return stats
    
    def reset(self):
        self.tracker = EnhancedByteTracker(RTMOTrackerConfig.get_config_dict())
        self.frame_count = 0
        self.stats = {'total_frames': 0, 'total_detections': 0}


class TrackingVisualizer:
    """트래킹 시각화"""
    def __init__(self, max_colors=50):
        self.colors = self.generate_colors(max_colors)
        
    def generate_colors(self, num_colors):
        colors = []
        for i in range(num_colors):
            hue = i / num_colors
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            colors.append(bgr)
        return colors
    
    def get_track_color(self, track_id):
        return self.colors[track_id % len(self.colors)]
    
    def draw_tracks(self, image, tracks, pose_result=None):
        vis_image = image.copy()
        
        for track in tracks:
            color = self.get_track_color(track.track_id)
            bbox = track.get_bbox()
            
            # 바운딩 박스
            cv2.rectangle(vis_image, (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])), color, 2)
            
            # 트랙 ID
            text = f"ID:{track.track_id} {track.score:.2f}"
            cv2.putText(vis_image, text, (int(bbox[0]), int(bbox[1] - 5)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return vis_image
    
    def draw_info_panel(self, image, frame_id, num_tracks, fps=None):
        info_text = [f"Frame: {frame_id}", f"Tracks: {num_tracks}"]
        if fps is not None:
            info_text.append(f"FPS: {fps:.1f}")
        
        # 배경 패널
        panel_height = len(info_text) * 30 + 10
        cv2.rectangle(image, (10, 10), (250, 10 + panel_height), (0, 0, 0), -1)
        cv2.rectangle(image, (10, 10), (250, 10 + panel_height), (255, 255, 255), 2)
        
        # 정보 텍스트
        for i, text in enumerate(info_text):
            cv2.putText(image, text, (20, 35 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return image


class VideoProcessor:
    """비디오 처리"""
    def __init__(self, pipeline, visualizer):
        self.pipeline = pipeline
        self.visualizer = visualizer
        
    def process_video(self, input_path, output_path, show_video=False, max_frames=None):
        print(f"Processing video: {input_path}")
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        # 비디오 정보
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames is not None:
            total_frames = min(total_frames, max_frames)
        
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {total_frames}")
        
        # 비디오 라이터
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        start_time = time.time()
        
        try:
            while frame_idx < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_start = time.time()
                
                # 프레임 처리
                tracks, pose_result = self.pipeline.process_frame(frame)
                
                # 시각화
                vis_frame = self.visualizer.draw_tracks(frame, tracks, pose_result)
                
                # 정보 패널
                current_fps = 1.0 / (time.time() - frame_start) if time.time() - frame_start > 0 else 0
                vis_frame = self.visualizer.draw_info_panel(vis_frame, frame_idx, len(tracks), current_fps)
                
                # 저장
                out.write(vis_frame)
                
                # 화면 표시
                if show_video:
                    cv2.imshow('RTMO Tracking', vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                frame_idx += 1
                
                # 진행상황 출력
                if frame_idx % 50 == 0:
                    elapsed = time.time() - start_time
                    avg_fps = frame_idx / elapsed
                    print(f"Processed {frame_idx}/{total_frames} frames, avg FPS: {avg_fps:.1f}")
        
        finally:
            cap.release()
            out.release()
            if show_video:
                cv2.destroyAllWindows()
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_fps = frame_idx / total_time if total_time > 0 else 0
        
        result = {
            'input_path': input_path,
            'output_path': output_path,
            'processed_frames': frame_idx,
            'total_time': total_time,
            'average_fps': avg_fps,
            'pipeline_stats': self.pipeline.get_stats()
        }
        
        print(f"Processing completed!")
        print(f"  Processed frames: {frame_idx}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average FPS: {avg_fps:.1f}")
        
        return result


# === 메인 함수들 ===
def parse_args():
    parser = argparse.ArgumentParser(description='Enhanced ByteTracker with RTMO Demo')
    
    parser.add_argument('--rtmo-config', 
                       default=RTMO_CONFIG,
                       help='RTMO config file path')
    parser.add_argument('--rtmo-checkpoint',
                       default=RTMO_CHECKPOINT, 
                       help='RTMO checkpoint file path')
    
    parser.add_argument('--input', '-i', help='Input video file path')
    parser.add_argument('--output-dir', '-o', default='./output', help='Output directory')
    
    parser.add_argument('--device', default='cuda:0', help='Device for inference')
    parser.add_argument('--show', action='store_true', help='Show video during processing')
    parser.add_argument('--max-frames', type=int, help='Maximum frames to process')
    
    parser.add_argument('--config-mode', 
                       choices=['default', 'rtmo', 'fast', 'accurate'],
                       default='rtmo',
                       help='Tracker configuration mode')
    
    return parser.parse_args()


def get_tracker_config(mode):
    if mode == 'default':
        return DefaultTrackerConfig.get_config_dict()
    elif mode == 'rtmo':
        return RTMOTrackerConfig.get_config_dict()
    elif mode == 'fast':
        config = RTMOTrackerConfig.get_config_dict()
        config['obj_score_thrs'] = {'high': 0.4, 'low': 0.1}
        config['num_tentatives'] = 1
        return config
    elif mode == 'accurate':
        config = RTMOTrackerConfig.get_config_dict()
        config['obj_score_thrs'] = {'high': 0.7, 'low': 0.2}
        config['num_tentatives'] = 4
        return config
    else:
        return RTMOTrackerConfig.get_config_dict()


def test_with_default_videos(args, pipeline, processor):
    test_videos = [
        '/workspace/rtmo_gcn_pipeline/rtmo_pose_track/raw_videos/Fight/cam04_06.mp4',
        '/workspace/rtmo_gcn_pipeline/rtmo_pose_track/raw_videos/Fight/F_4_0_0_0_0.mp4'
    ]
    
    results = []
    
    for video_path in test_videos:
        if not Path(video_path).exists():
            print(f"Warning: Video not found: {video_path}")
            continue
        
        video_name = Path(video_path).name
        print(f"\n{'='*60}")
        print(f"Processing: {video_name}")
        print(f"Path: {video_path}")
        print(f"{'='*60}")
        
        output_path = Path(args.output_dir) / f"tracked_{video_name}"
        
        try:
            pipeline.reset()
            Track.count = 0  # 트랙 ID 리셋
            
            result = processor.process_video(
                input_path=video_path,
                output_path=str(output_path),
                show_video=args.show,
                max_frames=args.max_frames
            )
            
            results.append({'video_name': video_name, 'result': result})
            print(f"✅ Completed: {video_name}")
            print(f"   Output: {output_path}")
            
        except Exception as e:
            print(f"❌ Error processing {video_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results


def print_summary(results):
    print(f"\n{'='*80}")
    print("🎯 PROCESSING SUMMARY")
    print(f"{'='*80}")
    
    total_frames = 0
    total_time = 0
    
    for result_info in results:
        video_name = result_info['video_name']
        result = result_info['result']
        
        print(f"\n📹 {video_name}")
        print(f"   Frames processed: {result['processed_frames']}")
        print(f"   Total time: {result['total_time']:.2f}s")
        print(f"   Average FPS: {result['average_fps']:.1f}")
        
        if 'pipeline_stats' in result:
            stats = result['pipeline_stats']
            print(f"   Total tracks: {stats.get('total_tracks', 'N/A')}")
        
        total_frames += result['processed_frames']
        total_time += result['total_time']
    
    if results:
        overall_fps = total_frames / total_time if total_time > 0 else 0
        print(f"\n📊 Overall Performance")
        print(f"   Total frames: {total_frames}")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Overall FPS: {overall_fps:.1f}")


def main():
    args = parse_args()
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Enhanced ByteTracker with RTMO Demo")
    print(f"📁 Output directory: {output_dir.absolute()}")
    print(f"🔧 Device: {args.device}")
    print(f"⚙️  Tracker config mode: {args.config_mode}")
    
    # 모델 파일 확인
    rtmo_config = Path(args.rtmo_config)
    rtmo_checkpoint = Path(args.rtmo_checkpoint)
    
    if not rtmo_config.exists():
        print(f"❌ Error: RTMO config file not found: {rtmo_config}")
        return
    
    if not rtmo_checkpoint.exists():
        print(f"❌ Error: RTMO checkpoint file not found: {rtmo_checkpoint}")
        return
    
    try:
        # 설정 가져오기
        tracker_config = get_tracker_config(args.config_mode)
        
        # 파이프라인 초기화
        print("\n🔄 Initializing RTMO tracking pipeline...")
        pipeline = RTMOTrackingPipeline(
            rtmo_config=str(rtmo_config),
            rtmo_checkpoint=str(rtmo_checkpoint),
            device=args.device,
            tracker_config=tracker_config
        )
        
        # 시각화기 및 프로세서 초기화
        visualizer = TrackingVisualizer()
        processor = VideoProcessor(pipeline, visualizer)
        
        print("✅ Pipeline initialized successfully!")
        
        if args.input:
            # 단일 비디오 처리
            input_path = Path(args.input)
            if not input_path.exists():
                print(f"❌ Error: Input video not found: {input_path}")
                return
            
            output_path = output_dir / f"tracked_{input_path.name}"
            print(f"\n🎬 Processing single video: {input_path}")
            
            result = processor.process_video(
                input_path=str(input_path),
                output_path=str(output_path),
                show_video=args.show,
                max_frames=args.max_frames
            )
            
            print_summary([{'video_name': input_path.name, 'result': result}])
        else:
            # 기본 테스트 비디오들로 테스트
            print(f"\n🎯 Testing with default videos...")
            results = test_with_default_videos(args, pipeline, processor)
            
            if results:
                print_summary(results)
            else:
                print("❌ No videos were successfully processed.")
        
        print(f"\n🎉 Demo completed! Check {output_dir} for results.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()