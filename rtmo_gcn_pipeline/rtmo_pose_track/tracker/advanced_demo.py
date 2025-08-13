#!/usr/bin/env python3
"""
Enhanced ByteTracker with RTMO - Advanced Demo with Tracking
실제 ByteTracker 알고리즘을 포함한 고급 데모
"""

import os
import sys
import cv2
import numpy as np
import time
import colorsys
from pathlib import Path
from enum import Enum
from typing import List, Tuple, Optional
from collections import deque

# 필요한 경로들을 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, '/workspace')

try:
    from mmpose.apis import init_model, inference_bottomup
    import torch
    from scipy.optimize import linear_sum_assignment
    print("Dependencies imported successfully!")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# RTMO 모델과 체크포인트 경로
RTMO_CONFIG = '/workspace/mmpose/configs/body_2d_keypoint/rtmo/body7/rtmo-m_16xb16-600e_body7-640x640.py'
RTMO_CHECKPOINT = '/workspace/mmpose/checkpoints/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.pth'

# 테스트 비디오 경로들
TEST_VIDEOS = [
    '/workspace/rtmo_gcn_pipeline/rtmo_pose_track/raw_videos/Fight/cam04_06.mp4',
    '/workspace/rtmo_gcn_pipeline/rtmo_pose_track/raw_videos/Fight/F_4_0_0_0_0.mp4'
]


class TrackState(Enum):
    """트랙 상태"""
    NEW = 1
    TRACKED = 2
    LOST = 3
    REMOVED = 4


class SimpleKalmanFilter:
    """간단한 칼만 필터 (2D 바운딩 박스용)"""
    def __init__(self):
        self.dt = 1.0
        # 상태: [cx, cy, w, h, vcx, vcy, vw, vh]
        self.state = np.zeros(8, dtype=np.float32)
        self.initialized = False
        
    def initiate(self, bbox):
        """바운딩 박스로 초기화: [x1, y1, x2, y2]"""
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        
        self.state = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32)
        self.initialized = True
        
    def predict(self):
        """예측 스텝"""
        if not self.initialized:
            return
        
        # 간단한 등속도 모델
        self.state[0] += self.state[4] * self.dt  # cx
        self.state[1] += self.state[5] * self.dt  # cy
        self.state[2] += self.state[6] * self.dt  # w
        self.state[3] += self.state[7] * self.dt  # h
        
    def update(self, bbox):
        """측정값으로 업데이트"""
        if not self.initialized:
            self.initiate(bbox)
            return
        
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        
        # 간단한 업데이트 (가중평균)
        alpha = 0.7
        self.state[4] = alpha * (cx - self.state[0]) + (1 - alpha) * self.state[4]  # vcx
        self.state[5] = alpha * (cy - self.state[1]) + (1 - alpha) * self.state[5]  # vcy
        self.state[6] = alpha * (w - self.state[2]) + (1 - alpha) * self.state[6]   # vw
        self.state[7] = alpha * (h - self.state[3]) + (1 - alpha) * self.state[7]   # vh
        
        self.state[0] = cx
        self.state[1] = cy
        self.state[2] = w
        self.state[3] = h
        
    def get_current_bbox(self):
        """현재 바운딩 박스 반환"""
        if not self.initialized:
            return np.array([0, 0, 0, 0])
        
        cx, cy, w, h = self.state[:4]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return np.array([x1, y1, x2, y2])


class Track:
    """간단한 트랙 클래스"""
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
        """예측"""
        self.kalman.predict()
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        
    def update(self, bbox, score):
        """업데이트"""
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
        """놓친 것으로 표시"""
        if self.state == TrackState.TRACKED:
            self.state = TrackState.LOST
            
    def get_bbox(self):
        """현재 바운딩 박스 반환"""
        return self.kalman.get_current_bbox()


class SimpleByteTracker:
    """간단한 ByteTracker 구현"""
    def __init__(self, high_thresh=0.6, low_thresh=0.1, iou_thresh=0.3, max_lost=30):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.iou_thresh = iou_thresh
        self.max_lost = max_lost
        
        self.tracked_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []
        self.frame_count = 0
        
    def compute_iou(self, bbox1, bbox2):
        """IoU 계산"""
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
    
    def associate_detections_to_trackers(self, detections, trackers):
        """헝가리안 알고리즘으로 매칭"""
        if len(trackers) == 0:
            return [], list(range(len(detections))), []
        if len(detections) == 0:
            return [], [], list(range(len(trackers)))
            
        # IoU 매트릭스 계산
        iou_matrix = np.zeros((len(detections), len(trackers)))
        for d, det in enumerate(detections):
            for t, tracker in enumerate(trackers):
                iou_matrix[d, t] = self.compute_iou(det[:4], tracker.get_bbox())
        
        # 헝가리안 알고리즘 적용 (최대화 문제를 최소화로 변환)
        cost_matrix = 1 - iou_matrix
        
        if min(cost_matrix.shape) > 0:
            det_indices, trk_indices = linear_sum_assignment(cost_matrix)
            matches = []
            for d, t in zip(det_indices, trk_indices):
                if iou_matrix[d, t] >= self.iou_thresh:
                    matches.append([d, t])
            
            unmatched_dets = [d for d in range(len(detections)) if d not in det_indices or 
                             iou_matrix[det_indices[list(det_indices).index(d)], 
                             trk_indices[list(det_indices).index(d)]] < self.iou_thresh]
            unmatched_trks = [t for t in range(len(trackers)) if t not in trk_indices or
                             iou_matrix[det_indices[list(trk_indices).index(t)], t] < self.iou_thresh]
        else:
            matches = []
            unmatched_dets = list(range(len(detections)))
            unmatched_trks = list(range(len(trackers)))
        
        return matches, unmatched_dets, unmatched_trks
    
    def update(self, detections):
        """프레임 업데이트"""
        self.frame_count += 1
        
        # 예측 스텝
        for track in self.tracked_tracks + self.lost_tracks:
            track.predict()
        
        # 높은 점수 detection 분리
        high_dets = detections[detections[:, 4] >= self.high_thresh] if len(detections) > 0 else []
        low_dets = detections[detections[:, 4] < self.high_thresh] if len(detections) > 0 else []
        low_dets = low_dets[low_dets[:, 4] >= self.low_thresh] if len(low_dets) > 0 else []
        
        # 첫 번째 매칭: 높은 점수 detection과 tracked tracks
        matches, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(
            high_dets, self.tracked_tracks)
        
        # 매칭된 트랙 업데이트
        for match in matches:
            det_idx, trk_idx = match
            self.tracked_tracks[trk_idx].update(high_dets[det_idx][:4], high_dets[det_idx][4])
        
        # 매칭되지 않은 tracked tracks를 lost로 이동
        for trk_idx in unmatched_trks:
            self.tracked_tracks[trk_idx].mark_missed()
        
        # lost tracks 리스트 업데이트
        self.lost_tracks.extend([self.tracked_tracks[i] for i in unmatched_trks])
        self.tracked_tracks = [self.tracked_tracks[i] for i in range(len(self.tracked_tracks)) 
                              if i not in unmatched_trks]
        
        # 두 번째 매칭: 낮은 점수 detection과 lost tracks
        if len(low_dets) > 0 and len(self.lost_tracks) > 0:
            matches2, unmatched_dets2, unmatched_trks2 = self.associate_detections_to_trackers(
                low_dets, self.lost_tracks)
            
            # lost tracks에서 재매칭된 것들을 tracked로 복귀
            for match in matches2:
                det_idx, trk_idx = match
                self.lost_tracks[trk_idx].update(low_dets[det_idx][:4], low_dets[det_idx][4])
                self.tracked_tracks.append(self.lost_tracks[trk_idx])
            
            # lost tracks 업데이트
            self.lost_tracks = [self.lost_tracks[i] for i in range(len(self.lost_tracks)) 
                               if i not in [match[1] for match in matches2]]
        
        # 새로운 트랙 생성 (높은 점수 unmatched detections)
        unmatched_high_dets = [high_dets[i] for i in unmatched_dets if i < len(high_dets)]
        for det in unmatched_high_dets:
            new_track = Track(det[:4], det[4])
            self.tracked_tracks.append(new_track)
        
        # 오래된 lost tracks 제거
        for track in self.lost_tracks[:]:
            if track.time_since_update > self.max_lost:
                self.lost_tracks.remove(track)
                self.removed_tracks.append(track)
        
        # 반환할 트랙들 (TRACKED 상태만)
        active_tracks = [t for t in self.tracked_tracks if t.state == TrackState.TRACKED]
        
        return active_tracks


def generate_colors(num_colors):
    """고유한 색상 생성"""
    colors = []
    for i in range(num_colors):
        hue = i / num_colors
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        colors.append(bgr)
    return colors


def draw_tracks(image, tracks):
    """트랙들을 이미지에 그리기"""
    colors = generate_colors(50)
    vis_image = image.copy()
    
    for track in tracks:
        bbox = track.get_bbox()
        color = colors[track.track_id % len(colors)]
        
        # 바운딩 박스
        cv2.rectangle(vis_image, (int(bbox[0]), int(bbox[1])), 
                     (int(bbox[2]), int(bbox[3])), color, 2)
        
        # 트랙 ID
        text = f"ID:{track.track_id} {track.score:.2f}"
        cv2.putText(vis_image, text, (int(bbox[0]), int(bbox[1] - 5)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return vis_image


def extract_detections_from_pose_result(pose_result):
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
    
    # [x1, y1, x2, y2, score] 형태로 결합
    detections = np.column_stack([bboxes, scores])
    return detections


def process_video_with_tracking(video_path, output_path, max_frames=200):
    """ByteTracker를 사용한 비디오 처리"""
    print(f"\nProcessing with ByteTracker: {Path(video_path).name}")
    
    # RTMO 모델 초기화
    print("Initializing RTMO model...")
    model = init_model(RTMO_CONFIG, RTMO_CHECKPOINT, device='cuda:0')
    
    # ByteTracker 초기화
    tracker = SimpleByteTracker(high_thresh=0.5, low_thresh=0.1, iou_thresh=0.3)
    
    # 비디오 열기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    # 비디오 정보
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    
    # 비디오 출력 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while frame_count < min(total_frames, max_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_start = time.time()
            
            # RTMO로 포즈 추정
            pose_results = inference_bottomup(model, frame)
            
            # detection 추출
            detections = np.empty((0, 5))
            if pose_results:
                detections = extract_detections_from_pose_result(pose_results[0])
            
            # ByteTracker로 추적
            active_tracks = tracker.update(detections)
            
            # 시각화
            vis_frame = draw_tracks(frame, active_tracks)
            
            # 정보 패널 추가
            info_text = f"Frame: {frame_count}, Tracks: {len(active_tracks)}, Detections: {len(detections)}"
            cv2.putText(vis_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # FPS 표시
            current_fps = 1.0 / (time.time() - frame_start) if time.time() - frame_start > 0 else 0
            fps_text = f"FPS: {current_fps:.1f}"
            cv2.putText(vis_frame, fps_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 비디오에 저장
            out.write(vis_frame)
            
            frame_count += 1
            
            # 진행상황 출력
            if frame_count % 50 == 0:
                elapsed = time.time() - start_time
                avg_fps = frame_count / elapsed
                print(f"  Processed {frame_count}/{min(total_frames, max_frames)} frames, "
                      f"avg FPS: {avg_fps:.1f}, active tracks: {len(active_tracks)}")
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_fps = frame_count / total_time
        
        print(f"  Completed! Processed {frame_count} frames in {total_time:.2f}s")
        print(f"  Average FPS: {avg_fps:.1f}")
        print(f"  Total tracks created: {Track.count}")
        print(f"  Output saved to: {output_path}")
        
    except Exception as e:
        print(f"  Error during processing: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        cap.release()
        out.release()


def main():
    """메인 함수"""
    print("Enhanced ByteTracker with RTMO - Advanced Demo")
    print("=" * 60)
    
    # 출력 디렉토리 생성
    output_dir = Path('./advanced_output')
    output_dir.mkdir(exist_ok=True)
    
    print(f"Output directory: {output_dir.absolute()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # 각 테스트 비디오 처리
    for video_path in TEST_VIDEOS:
        if Path(video_path).exists():
            video_name = Path(video_path).name
            output_path = str(output_dir / f"tracked_{video_name}")
            
            try:
                Track.count = 0  # 트랙 ID 리셋
                process_video_with_tracking(video_path, output_path, max_frames=300)
            except Exception as e:
                print(f"Failed to process {video_name}: {e}")
                continue
        else:
            print(f"Warning: Video not found: {video_path}")
    
    print("\nAdvanced demo completed!")
    print("Check the advanced_output directory for tracking results with Track IDs!")


if __name__ == '__main__':
    main()