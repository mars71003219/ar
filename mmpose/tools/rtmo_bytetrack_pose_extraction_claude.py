# Copyright (c) OpenMMLab. All rights reserved.
import os
# import debugpy
# debugpy.listen(("0.0.0.0", 5678)) # 컨테이너 내부에서 모든 IP로부터 접속 허용
# print(f'debugpy ready..')
# debugpy.wait_for_client() # VS Code 디버거가 연결될 때까지 코드 실행을 일시 중지
# print(f"Current working directory inside container: {os.getcwd()}")
# print(f"Script path inside container: {os.path.abspath(__file__)}")

import logging
import mimetypes
import glob
import time
from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import mmcv
import mmengine
import numpy as np
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

from mmpose.apis import inference_bottomup, init_model
from mmpose.registry import VISUALIZERS


class KalmanFilter:
    """2D 바운딩 박스 트래킹을 위한 간단한 칼만 필터"""
    
    def __init__(self):
        # 상태: [center_x, center_y, width, height, dx, dy, dw, dh]
        self.state = np.zeros(8)
        
        # 상태 전이 행렬 (위치와 크기, 그리고 속도)
        self.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],  # x = x + dx
            [0, 1, 0, 0, 0, 1, 0, 0],  # y = y + dy
            [0, 0, 1, 0, 0, 0, 1, 0],  # w = w + dw
            [0, 0, 0, 1, 0, 0, 0, 1],  # h = h + dh
            [0, 0, 0, 0, 1, 0, 0, 0],  # dx = dx
            [0, 0, 0, 0, 0, 1, 0, 0],  # dy = dy
            [0, 0, 0, 0, 0, 0, 1, 0],  # dw = dw
            [0, 0, 0, 0, 0, 0, 0, 1],  # dh = dh
        ])
        
        # 관측 행렬 (관측하는 것은 center_x, center_y, width, height)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ])
        
        # 공분산 행렬
        self.P = np.eye(8) * 1000  # 초기 불확실성
        
        # 프로세스 노이즈
        self.Q = np.eye(8)
        self.Q[:4, :4] *= 1.0  # 위치/크기 노이즈
        self.Q[4:, 4:] *= 1.0   # 속도 노이즈
        
        # 관측 노이즈
        self.R = np.eye(4) * 10.0
        
        self.initialized = False
    
    def init_state(self, bbox):
        """바운딩 박스로 상태 초기화"""
        center_x = (bbox[0] + bbox[2]) / 2.0
        center_y = (bbox[1] + bbox[3]) / 2.0
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        self.state[:4] = [center_x, center_y, width, height]
        self.state[4:] = 0.0  # 초기 속도는 0
        self.initialized = True
    
    def predict(self):
        """예측 단계"""
        if not self.initialized:
            return self.get_bbox()
            
        # 상태 예측
        self.state = self.F @ self.state
        
        # 공분산 예측
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.get_bbox()
    
    def update(self, bbox):
        """업데이트 단계"""
        if not self.initialized:
            self.init_state(bbox)
            return self.get_bbox()
        
        # 관측값 (center_x, center_y, width, height)
        center_x = (bbox[0] + bbox[2]) / 2.0
        center_y = (bbox[1] + bbox[3]) / 2.0
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        z = np.array([center_x, center_y, width, height])
        
        # 잔차 계산
        y = z - self.H @ self.state
        
        # 잔차 공분산
        S = self.H @ self.P @ self.H.T + self.R
        
        # 칼만 이득
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # 상태 업데이트
        self.state = self.state + K @ y
        
        # 공분산 업데이트
        I_KH = np.eye(8) - K @ self.H
        self.P = I_KH @ self.P
        
        return self.get_bbox()
    
    def get_bbox(self):
        """현재 상태에서 바운딩 박스 반환"""
        if not self.initialized:
            return np.array([0, 0, 0, 0])
            
        center_x, center_y, width, height = self.state[:4]
        
        # 음수 크기 방지
        width = max(width, 1.0)
        height = max(height, 1.0)
        
        x1 = center_x - width / 2.0
        y1 = center_y - height / 2.0
        x2 = center_x + width / 2.0
        y2 = center_y + height / 2.0
        
        return np.array([x1, y1, x2, y2])

@dataclass
class Track:
    """트랙 객체 (칼만 필터 포함)"""
    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    score: float
    age: int = 0
    hits: int = 1
    hit_streak: int = 1
    time_since_update: int = 0
    
    def __post_init__(self):
        """칼만 필터 초기화"""
        self.kalman = KalmanFilter()
        self.kalman.init_state(self.bbox)
    
    def update(self, bbox: np.ndarray, score: float):
        """트랙 업데이트 (칼만 필터 사용)"""
        self.bbox = self.kalman.update(bbox)
        self.score = score
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0
    
    def predict(self):
        """예측 단계 (칼만 필터 사용)"""
        self.bbox = self.kalman.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

class ByteTracker:
    """ByteTrack 알고리즘 구현 (칼만 필터 포함)"""
    
    def __init__(self, high_thresh: float = 0.6, low_thresh: float = 0.1, 
                 max_disappeared: int = 30, min_hits: int = 3):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.max_disappeared = max_disappeared
        self.min_hits = min_hits
        
        self.tracks: List[Track] = []
        self.next_id = 0
        
    def update(self, detections: np.ndarray) -> List[Track]:
        """
        detections: [N, 5] array of [x1, y1, x2, y2, score]
        Returns: List of active tracks
        """
        # Predict existing tracks
        for track in self.tracks:
            track.predict()
        
        # Separate detections by confidence
        high_dets = detections[detections[:, 4] >= self.high_thresh]
        low_dets = detections[(detections[:, 4] >= self.low_thresh) & 
                             (detections[:, 4] < self.high_thresh)]
        
        # First association with high confidence detections
        matched_tracks, unmatched_dets, unmatched_tracks = self._associate(
            self.tracks, high_dets, iou_threshold=0.5)
        
        # Update matched tracks
        for track_idx, det_idx in matched_tracks:
            self.tracks[track_idx].update(high_dets[det_idx, :4], high_dets[det_idx, 4])
        
        # Second association with low confidence detections for unmatched tracks
        unmatched_tracks_for_low = [self.tracks[i] for i in unmatched_tracks 
                                   if self.tracks[i].time_since_update == 1]
        
        if len(unmatched_tracks_for_low) > 0 and len(low_dets) > 0:
            matched_tracks_low, unmatched_dets_low, unmatched_tracks_low = self._associate(
                unmatched_tracks_for_low, low_dets, iou_threshold=0.5)
            
            # Update matched tracks from low confidence detections
            for track_idx, det_idx in matched_tracks_low:
                track = unmatched_tracks_for_low[track_idx]
                track.update(low_dets[det_idx, :4], low_dets[det_idx, 4])
        
        # Create new tracks for unmatched high confidence detections
        for det_idx in unmatched_dets:
            new_track = Track(
                track_id=self.next_id,
                bbox=high_dets[det_idx, :4],
                score=high_dets[det_idx, 4]
            )
            self.tracks.append(new_track)
            self.next_id += 1
        
        # Remove tracks that have been unmatched for too long
        self.tracks = [track for track in self.tracks 
                      if track.time_since_update < self.max_disappeared]
        
        # Return tracks that meet minimum hit requirements
        active_tracks = [track for track in self.tracks 
                        if track.hits >= self.min_hits or track.time_since_update < 1]
        
        return active_tracks

    def _associate(self, tracks: List[Track], detections: np.ndarray, 
                  iou_threshold: float = 0.5) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """IoU 기반 연관"""
        if len(tracks) == 0:
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            return [], [], list(range(len(tracks)))
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(tracks), len(detections)))
        for t, track in enumerate(tracks):
            for d, det in enumerate(detections):
                iou_matrix[t, d] = self._compute_iou(track.bbox, det[:4])
        
        # Hungarian algorithm for optimal assignment
        row_indices, col_indices = linear_sum_assignment(-iou_matrix)
        
        matched_tracks = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(range(len(tracks)))
        
        for row, col in zip(row_indices, col_indices):
            if iou_matrix[row, col] >= iou_threshold:
                matched_tracks.append((row, col))
                unmatched_dets.remove(col)
                unmatched_tracks.remove(row)
        
        return matched_tracks, unmatched_dets, unmatched_tracks

    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """두 바운딩 박스의 IoU 계산"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0.0

def create_detection_results(pose_result):
    """
    포즈 추정 결과에서 detection 형태로 변환
    """
    pred_instances = pose_result._pred_instances
    
    if not hasattr(pred_instances, 'bboxes') or len(pred_instances.bboxes) == 0:
        return np.empty((0, 5))
    
    bboxes = pred_instances.bboxes.cpu().numpy() if hasattr(pred_instances.bboxes, 'cpu') else pred_instances.bboxes
    if hasattr(pred_instances, 'bbox_scores'):
        scores = pred_instances.bbox_scores.cpu().numpy() if hasattr(pred_instances.bbox_scores, 'cpu') else pred_instances.bbox_scores
        scores = scores.reshape(-1, 1)
    else:
        scores = np.ones((len(bboxes), 1))
    
    # [x1, y1, x2, y2, score] 형태로 반환 (이미 model config에서 threshold 적용됨)
    detections = np.concatenate([bboxes, scores], axis=1)
    return detections

def assign_track_ids_from_bytetrack(pose_result, active_tracks, iou_threshold=0.5):
    """
    ByteTrack 결과를 기반으로 pose_result에 track_id 할당
    """
    frame_result = pose_result._pred_instances
    if not hasattr(frame_result, 'bboxes') or len(frame_result.bboxes) == 0:
        frame_result.track_ids = np.array([])
        return pose_result

    pose_bboxes = frame_result.bboxes.cpu().numpy() if hasattr(frame_result.bboxes, 'cpu') else frame_result.bboxes
    track_ids = np.full(len(pose_bboxes), -1, dtype=int)
    
    # Active tracks와 pose bboxes 매칭
    for i, pose_bbox in enumerate(pose_bboxes):
        best_iou = 0
        best_track_id = -1
        
        for track in active_tracks:
            current_iou = calculate_iou(pose_bbox, track.bbox)
            if current_iou > best_iou and current_iou > iou_threshold:
                best_iou = current_iou
                best_track_id = track.track_id
        
        track_ids[i] = best_track_id

    frame_result.track_ids = track_ids
    return pose_result

def calculate_iou(box1, box2):
    """두 바운딩 박스의 IoU 계산"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0.0

def draw_tracking_info(img, results, track_ids):
    """트래킹 ID를 이미지에 그리기"""
    if not hasattr(results, 'bboxes') or len(track_ids) == 0:
        return img
    
    bboxes = results.bboxes.cpu().numpy() if hasattr(results.bboxes, 'cpu') else results.bboxes
    
    for i, track_id in enumerate(track_ids):
        if track_id >= 0 and i < len(bboxes):
            bbox = bboxes[i]
            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int(bbox[1] - 10)
            
            cv2.putText(img, f'ID:{track_id}', (center_x, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return img

def get_num_keypoints_from_model(pose_model):
    """모델 설정에서 키포인트 개수를 가져오기"""
    try:
        # 모델의 dataset_meta에서 키포인트 개수 추출
        if hasattr(pose_model, 'dataset_meta') and pose_model.dataset_meta is not None:
            if 'num_keypoints' in pose_model.dataset_meta:
                return pose_model.dataset_meta['num_keypoints']
            elif 'keypoint_info' in pose_model.dataset_meta:
                return len(pose_model.dataset_meta['keypoint_info'])
            elif 'keypoints' in pose_model.dataset_meta:
                return len(pose_model.dataset_meta['keypoints'])
        
        # 모델 config에서 추출 시도
        if hasattr(pose_model, 'cfg'):
            if hasattr(pose_model.cfg, 'model') and hasattr(pose_model.cfg.model, 'num_keypoints'):
                return pose_model.cfg.model.num_keypoints
            elif hasattr(pose_model.cfg, 'num_keypoints'):
                return pose_model.cfg.num_keypoints
    except Exception as e:
        print(f"Warning: Could not extract keypoint number from model: {e}")
    
    # 기본값으로 17 (COCO format) 반환
    print("Warning: Using default keypoint number (17). Consider checking your model configuration.")
    return 17

def create_tracked_pose_annotation(pose_results, video_path, pose_model, n_person=2):
    """
    트래킹 결과를 기반으로 시간적 연속성을 보장하는 annotation을 생성합니다.
    """
    if not pose_results:
        return None

    # 1. 모델에서 키포인트 개수 추출
    num_keypoints = get_num_keypoints_from_model(pose_model)
    print(f"Using {num_keypoints} keypoints from model configuration")

    # 2. 모든 프레임에서 유효한 track_id 수집
    all_track_ids = set()
    for result in pose_results:
        if hasattr(result._pred_instances, 'track_ids'):
            for tid in result._pred_instances.track_ids:
                if tid >= 0:
                    all_track_ids.add(tid)
    
    if not all_track_ids:
        print("Warning: No valid tracks found. Cannot create annotation.")
        return None

    # 3. 가장 오래 지속된 track_id를 n_person 만큼 선택
    track_lengths = {tid: 0 for tid in all_track_ids}
    for result in pose_results:
        if hasattr(result._pred_instances, 'track_ids'):
            for tid in result._pred_instances.track_ids:
                if tid in track_lengths:
                    track_lengths[tid] += 1
    
    # 길이를 기준으로 정렬하여 상위 n_person개의 ID 선택
    sorted_tracks = sorted(track_lengths.items(), key=lambda item: item[1], reverse=True)
    final_track_ids = [item[0] for item in sorted_tracks[:n_person]]
    
    print(f"Selected track IDs: {final_track_ids} (lengths: {[track_lengths[tid] for tid in final_track_ids]})")
    
    # track_id를 배열 인덱스(0, 1, ...)로 매핑
    track_id_to_idx = {tid: i for i, tid in enumerate(final_track_ids)}
    
    num_persons = len(final_track_ids)
    num_frames = len(pose_results)

    # 4. 최종 배열 초기화
    keypoints = np.zeros((num_persons, num_frames, num_keypoints, 2), dtype=np.float32)
    scores = np.zeros((num_persons, num_frames, num_keypoints), dtype=np.float32)

    # 5. track_id를 기준으로 데이터 채우기
    for f_idx, result in enumerate(pose_results):
        pred_instances = result._pred_instances
        if not hasattr(pred_instances, 'track_ids'):
            continue
            
        instance_track_ids = pred_instances.track_ids
        instance_keypoints = pred_instances.keypoints
        instance_scores = pred_instances.keypoint_scores

        for p_idx in range(len(instance_track_ids)):
            tid = instance_track_ids[p_idx]
            # 선택된 track_id인 경우에만 데이터 저장
            if tid in track_id_to_idx:
                person_idx = track_id_to_idx[tid]
                # 키포인트 개수 확인 및 조정
                available_kpts = min(instance_keypoints.shape[1], num_keypoints)
                keypoints[person_idx, f_idx, :available_kpts] = instance_keypoints[p_idx, :available_kpts]
                scores[person_idx, f_idx, :available_kpts] = instance_scores[p_idx, :available_kpts]

    # 6. 빈 프레임 보간 (간단한 이전 프레임 복사 방식)
    for p_idx in range(num_persons):
        for f_idx in range(1, num_frames):
            # 현재 프레임의 포즈 신뢰도가 매우 낮으면 이전 프레임 값으로 채움
            if scores[p_idx, f_idx].mean() < 0.1:
                keypoints[p_idx, f_idx] = keypoints[p_idx, f_idx - 1]
                scores[p_idx, f_idx] = scores[p_idx, f_idx - 1]

    return {
        'keypoint': keypoints,
        'keypoint_score': scores,
        'frame_dir': os.path.splitext(os.path.basename(video_path))[0],
        'img_shape': pose_results[0].img_shape,
        'original_shape': pose_results[0].ori_shape,
        'total_frames': num_frames,
        'num_keypoints': num_keypoints,
        'selected_track_ids': final_track_ids,
        'label': 1 if video_path.split('/')[-2] == 'fight' else 0
    }

def parse_args():
    parser = ArgumentParser(description='RTMO 포즈 추정 및 ByteTrack을 이용한 트래킹')
    parser.add_argument('config', help='RTMO config file')
    parser.add_argument('checkpoint', help='RTMO checkpoint file')
    parser.add_argument('--input', type=str,
                       default='/aivanas/raw/surveillance/action/violence/action_recognition/data/RWF-2000',
                       help='Video file path or directory containing videos')  # 수정된 부분
    parser.add_argument('--output-root', type=str,
                       default='/workspace/mmpose/output', help='Output directory')
    parser.add_argument('--save-annotation', action='store_true', default=True, 
                       help='Save pose annotation file')
    parser.add_argument('--save-overlayfile', action='store_true', default=True,
                       help='Save overlay visualization video file')  # 새로 추가
    parser.add_argument('--n-person', type=int, default=2, 
                       help='Number of main actors to track for annotation')
    parser.add_argument('--show', action='store_true', default=False, help='Show real-time visualization')
    parser.add_argument('--device', default='cuda:0', help='Device for inference')
    parser.add_argument('--kpt-thr', type=float, default=0.9, help='Keypoint score threshold')
    parser.add_argument('--radius', type=int, default=2, help='Keypoint radius')
    parser.add_argument('--thickness', type=int, default=1, help='Link thickness')
    parser.add_argument('--show-interval', type=float, default=0, help='Sleep seconds per frame')
    parser.add_argument('--score-thr', type=float, default=0.3, help='Detection score threshold')
    parser.add_argument('--nms-thr', type=float, default=0.35, help='NMS threshold')
    # ByteTrack 파라미터
    parser.add_argument('--track-high-thresh', type=float, default=0.6, help='High threshold for tracking')
    parser.add_argument('--track-low-thresh', type=float, default=0.1, help='Low threshold for tracking')
    parser.add_argument('--track-max-disappeared', type=int, default=30, help='Max frames a track can disappear')
    parser.add_argument('--track-min-hits', type=int, default=3, help='Min hits before a track is confirmed')
    return parser.parse_args()

def get_video_extension(video_path):
    """비디오 파일의 확장자 반환"""
    _, ext = os.path.splitext(video_path)
    return ext.lower()

# 비디오 파일 찾기 함수 추가 (main 함수 전에 추가)
def find_video_files(input_path):
    """입력 경로에서 모든 mp4 파일을 재귀적으로 찾기"""
    if os.path.isfile(input_path):
        return [input_path]
    
    video_extensions = ['*.mp4', '*.MP4', '*.avi', '*.AVI', '*.mov', '*.MOV']
    video_files = []
    
    for ext in video_extensions:
        pattern = os.path.join(input_path, '**', ext)
        video_files.extend(glob.glob(pattern, recursive=True))
    
    return sorted(video_files)

def get_output_path(video_path, input_root, output_root, extension):
    """입력 구조를 유지하면서 출력 경로 생성 - input의 마지막 폴더부터 시작"""
    # input_root의 마지막 폴더명 추출
    input_basename = os.path.basename(input_root.rstrip('/'))
    
    # video_path에서 input_basename 이후의 상대 경로 계산
    abs_video_path = os.path.abspath(video_path)
    abs_input_root = os.path.abspath(input_root)
    
    # input_root로부터의 상대 경로
    rel_path = os.path.relpath(abs_video_path, abs_input_root)
    # input_basename을 포함한 전체 경로 구성
    rel_path_with_base = os.path.join(input_basename, rel_path)
    
    # 파일명에서 확장자 제거하고 새 확장자 적용
    base_name = os.path.splitext(rel_path_with_base)[0]
    output_file = base_name + extension
    output_path = os.path.join(output_root, output_file)
    
    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    return output_path

def main():
    args = parse_args()
    
    if args.output_root:
        mmengine.mkdir_or_exist(args.output_root)
    
    # 비디오 파일 목록 수집
    video_files = find_video_files(args.input)
    if not video_files:
        print(f"No video files found in {args.input}")
        return
    
    print(f"Found {len(video_files)} video files to process")
    
    # RTMO 포즈 추정 모델 초기화
    pose_model = init_model(args.config, args.checkpoint, device=args.device)
    # 모델 config에 threshold 설정 적용
    if hasattr(pose_model.cfg, 'model'):
        if hasattr(pose_model.cfg.model, 'test_cfg'):
            pose_model.cfg.model.test_cfg.score_thr = args.score_thr
            pose_model.cfg.model.test_cfg.nms_thr = args.nms_thr
        else:
            pose_model.cfg.model.test_cfg = dict(score_thr=args.score_thr, nms_thr=args.nms_thr)
    
    if hasattr(pose_model, 'head') and hasattr(pose_model.head, 'test_cfg'):
        pose_model.head.test_cfg.score_thr = args.score_thr
        pose_model.head.test_cfg.nms_thr = args.nms_thr
    
    # 시각화 모델 초기화
    visualizer = VISUALIZERS.build(pose_model.cfg.visualizer)
    visualizer.set_dataset_meta(pose_model.dataset_meta, skeleton_style='mmpose')
    
    # 각 비디오 파일 처리
    for video_idx, video_path in enumerate(video_files):
        print(f"\nProcessing video {video_idx + 1}/{len(video_files)}: {video_path}")
        
        # ByteTracker 초기화 (각 비디오마다 새로 초기화)
        tracker = ByteTracker(
            high_thresh=args.track_high_thresh,
            low_thresh=args.track_low_thresh,
            max_disappeared=args.track_max_disappeared,
            min_hits=args.track_min_hits
        )
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f'Cannot open video file: {video_path}')
            continue
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        pose_results = []
        all_tracks = []
        frames_for_overlay = []  # 오버레이 비디오용 프레임

        # Step 1: 포즈 추정 및 트래킹
        print("Step 1: Running pose estimation and tracking...")
        start_time = time.time()
        frame_count = 0
        pbar = tqdm(total=total_frames, desc="Processing frames")

        while True:
            success, frame = cap.read()
            if not success:
                break

            # 포즈 추정
            batch_pose_results = inference_bottomup(pose_model, frame)
            pose_result = batch_pose_results[0]
            
            # Detection 결과 생성
            detections = create_detection_results(pose_result)
            
            # ByteTrack으로 트래킹 수행
            active_tracks = tracker.update(detections)
            
            # 포즈 결과에 트래킹 ID 할당
            pose_result = assign_track_ids_from_bytetrack(pose_result, active_tracks)
            
            pose_results.append(pose_result)
            all_tracks.append(active_tracks.copy())
            
            # 오버레이 비디오를 위해 프레임 저장
            if args.save_overlayfile:
                frames_for_overlay.append(frame.copy())

            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            pbar.set_postfix({'FPS': f'{fps:.2f}', 'Active Tracks': len(active_tracks)})
            pbar.update(1)

        cap.release()
        pbar.close()
        elapsed = time.time() - start_time
        fps = len(pose_results) / elapsed if elapsed > 0 else 0
        print(f"Processing completed. Average FPS: {fps:.2f}")

        # Step 2: 어노테이션 저장
        if args.save_annotation:
            print("Step 2: Creating and saving tracked pose annotation...")
            annotation = create_tracked_pose_annotation(pose_results, video_path, pose_model, n_person=args.n_person)
            
            if annotation is not None:
                # 출력 경로 계산 (폴더 구조 유지)
                if os.path.isfile(args.input):
                    input_root = os.path.dirname(args.input)
                else:
                    input_root = args.input
                
                pkl_output_path = get_output_path(video_path, input_root, args.output_root, '_rtmo_bytetrack_pose.pkl')
                mmengine.dump(annotation, pkl_output_path)
                print(f'Pose annotation saved to: {pkl_output_path}')
                print(f'Annotation contains {annotation["num_keypoints"]} keypoints for {len(annotation["selected_track_ids"])} persons')
            else:
                print("Warning: Could not create pose annotation.")

        # Step 3: 오버레이 비디오 저장
        if args.save_overlayfile:
            print("Step 3: Creating overlay visualization video...")
            
            # 출력 비디오 경로 계산
            if os.path.isfile(args.input):
                input_root = os.path.dirname(args.input)
            else:
                input_root = args.input
            
            # 입력 파일의 확장자 가져오기
            input_ext = get_video_extension(video_path)
            
            # 확장자별 코덱 및 출력 파일 설정
            if input_ext == '.mp4':
                # MP4 입력 -> MP4 출력
                try:
                    fourcc = cv2.VideoWriter_fourcc(*'H264')
                    video_output_path = get_output_path(video_path, input_root, args.output_root, '_rtmo_bytetrack_overlay.mp4')
                    out_writer = cv2.VideoWriter(video_output_path, fourcc, video_fps, (width, height), True)
                    
                    test_frame = np.zeros((height, width, 3), dtype=np.uint8)
                    success = out_writer.write(test_frame)
                    if not success:
                        raise Exception("H264 failed")
                except:
                    try:
                        fourcc = cv2.VideoWriter_fourcc(*'avc1')
                        out_writer = cv2.VideoWriter(video_output_path, fourcc, video_fps, (width, height))
                    except:
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out_writer = cv2.VideoWriter(video_output_path, fourcc, video_fps, (width, height))
                        
            elif input_ext == '.avi':
                # AVI 입력 -> AVI 출력
                try:
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    video_output_path = get_output_path(video_path, input_root, args.output_root, '_rtmo_bytetrack_overlay.avi')
                    out_writer = cv2.VideoWriter(video_output_path, fourcc, video_fps, (width, height))
                except:
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    out_writer = cv2.VideoWriter(video_output_path, fourcc, video_fps, (width, height))
                    
            elif input_ext == '.mov':
                # MOV 입력 -> MOV 출력
                try:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_output_path = get_output_path(video_path, input_root, args.output_root, '_rtmo_bytetrack_overlay.mov')
                    out_writer = cv2.VideoWriter(video_output_path, fourcc, video_fps, (width, height))
                except:
                    fourcc = cv2.VideoWriter_fourcc(*'H264')
                    out_writer = cv2.VideoWriter(video_output_path, fourcc, video_fps, (width, height))
            else:
                # 기타 확장자는 AVI로 fallback
                print(f"Warning: Unsupported input format {input_ext}, using AVI output")
                try:
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    video_output_path = get_output_path(video_path, input_root, args.output_root, '_rtmo_bytetrack_overlay.avi')
                    out_writer = cv2.VideoWriter(video_output_path, fourcc, video_fps, (width, height))
                except:
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    out_writer = cv2.VideoWriter(video_output_path, fourcc, video_fps, (width, height))
            
            print(f"Input format: {input_ext}, Output: {video_output_path}")
            vis_pbar = tqdm(total=len(pose_results), desc="Creating overlay video")
            
            for idx, (pose_result, frame) in enumerate(zip(pose_results, frames_for_overlay)):
                # 포즈 시각화
                visualizer.add_datasample(
                    'result',
                    frame,
                    data_sample=pose_result,
                    draw_gt=False,
                    draw_bbox=True,
                    draw_heatmap=False,
                    show_kpt_idx=False,
                    show=False,
                    wait_time=0,
                    kpt_thr=args.kpt_thr,
                )
                
                # 트래킹 ID 추가
                vis_img = draw_tracking_info(visualizer.get_image(),
                                           pose_result.pred_instances,
                                           pose_result.pred_instances.track_ids)
                
                # 트래킹 통계 정보 표시
                if idx < len(all_tracks):
                    active_count = len(all_tracks[idx])
                    cv2.putText(vis_img, f'Active Tracks: {active_count}', (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                out_writer.write(vis_img)
                vis_pbar.update(1)
            
            out_writer.release()
            vis_pbar.close()
            print(f"Overlay video saved to: {video_output_path}")

        # Step 4: 실시간 시각화 표시
        if args.show:
            print("Step 4: Showing real-time visualization...")
            cap = cv2.VideoCapture(video_path)
            wait_time = int(1000 / video_fps) if video_fps > 0 else 33  # 기본 30fps
            
            for idx, pose_result in enumerate(pose_results):
                success, frame = cap.read()
                if not success:
                    break
                
                # 포즈 시각화
                visualizer.add_datasample(
                    'result',
                    frame,
                    data_sample=pose_result,
                    draw_gt=False,
                    draw_bbox=True,
                    draw_heatmap=False,
                    show_kpt_idx=False,
                    show=False,
                    wait_time=0,
                    kpt_thr=args.kpt_thr,
                    radius=args.radius,
                    thickness=args.thickness
                )
                
                # 트래킹 ID 추가
                vis_img = draw_tracking_info(visualizer.get_image(),
                                             pose_result.pred_instances,
                                             pose_result.pred_instances.track_ids)
                
                # 트래킹 통계 정보 표시
                if idx < len(all_tracks):
                    active_count = len(all_tracks[idx])
                    cv2.putText(vis_img, f'Active Tracks: {active_count}', (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('RTMO + ByteTrack', vis_img)
                wait_key_time = max(1, int(wait_time + args.show_interval * 1000))
                if cv2.waitKey(wait_key_time) & 0xFF == 27:  # ESC 키로 종료
                    break
            
            cap.release()
            cv2.destroyAllWindows()
        
    print("All videos processed!")

if __name__ == '__main__':
    main()