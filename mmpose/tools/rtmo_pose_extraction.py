# Copyright (c) OpenMMLab. All rights reserved.
import logging
import mimetypes
import os
import time
from argparse import ArgumentParser
from collections import defaultdict

import cv2
import mmcv
import mmengine
import numpy as np
# tqdm 라이브러리 import 추가
from tqdm import tqdm

from mmpose.apis import inference_bottomup, init_model
from mmpose.registry import VISUALIZERS

# ----- 기존 유틸리티 함수 (intersection, iou, area 등)는 변경 없이 그대로 사용합니다 -----
def intersection(b0, b1):
    """두 바운딩 박스의 교집합 영역 계산"""
    l, r = max(b0[0], b1[0]), min(b0[2], b1[2])
    u, d = max(b0[1], b1[1]), min(b0[3], b1[3])
    return max(0, r - l) * max(0, d - u)

def iou(b0, b1):
    """두 바운딩 박스의 IoU 계산"""
    i = intersection(b0, b1)
    u = area(b0) + area(b1) - i
    return i / u if u > 0 else 0

def area(b):
    """바운딩 박스 면적 계산"""
    return (b[2] - b[0]) * (b[3] - b[1])

def bbox2tracklet(bbox_list, iou_thre=0.6, min_bbox_score=0.3):
    """바운딩 박스를 트래킹하여 tracklet 생성 (신뢰도 필터링 추가)"""
    tracklet_id = -1
    tracklet_st_frame = {}
    tracklets = defaultdict(list)
    
    for t, frame_bboxes in enumerate(bbox_list):
        # 신뢰도 낮은 bbox 필터링
        high_conf_bboxes = frame_bboxes[frame_bboxes[:, 4] >= min_bbox_score]
        
        for idx in range(high_conf_bboxes.shape[0]):
            bbox = high_conf_bboxes[idx]
            matched = False
            
            # 기존 tracklet과 매칭 시도
            for tlet_id in range(tracklet_id, -1, -1):
                if not tracklets[tlet_id]:
                    continue
                
                last_bbox = tracklets[tlet_id][-1][1]
                # 시간적 근접성과 IoU를 함께 고려
                cond1 = iou(last_bbox, bbox) >= iou_thre
                cond2 = (t - tracklets[tlet_id][-1][0]) < 10 # 마지막 탐지 후 10프레임 이내
                
                if cond1 and cond2:
                    matched = True
                    tracklets[tlet_id].append((t, bbox))
                    break
            
            # 매칭되지 않으면 새 tracklet 생성
            if not matched:
                tracklet_id += 1
                tracklet_st_frame[tracklet_id] = t
                tracklets[tracklet_id].append((t, bbox))
    
    return tracklets

def filter_tracklets(tracklets, min_length=10, min_avg_area=2000):
    """짧거나 작은 tracklet 제거"""
    if not tracklets:
        return {}

    def mean_area(track):
        boxes = np.stack([x[1] for x in track]).astype(np.float32)
        areas = (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])
        return np.mean(areas)

    return {
        k: v for k, v in tracklets.items()
        if len(v) >= min_length and mean_area(v) > min_avg_area
    }
    
# ----- 트래킹 및 시각화 관련 함수들은 큰 변경 없이 유지 -----
def assign_track_ids(pose_result, frame_assignments, frame_idx):
    frame_result = pose_result._pred_instances
    if not hasattr(frame_result, 'bboxes'):
        frame_result.track_ids = np.array([])
        return pose_result

    bboxes = frame_result.bboxes
    track_ids = np.full(len(bboxes), -1, dtype=int)
    
    if frame_idx in frame_assignments:
        assignments = frame_assignments[frame_idx]
        
        for bbox_idx in range(len(bboxes)):
            bbox = bboxes[bbox_idx]
            best_iou = 0.3 # IoU 임계값
            best_id = -1
            
            for track_id, track_bbox in assignments:
                current_iou = iou(bbox, track_bbox[:4])
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_id = track_id
            
            track_ids[bbox_idx] = best_id
    
    frame_result.track_ids = track_ids
    return pose_result

def draw_tracking_info(img, results, track_ids):
    if not hasattr(results, 'bboxes') or len(track_ids) == 0:
        return img
    
    bboxes = results.bboxes
    
    for i, track_id in enumerate(track_ids):
        if track_id >= 0 and i < len(bboxes):
            bbox = bboxes[i]
            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int(bbox[1] - 10)
            
            cv2.putText(img, f'ID:{track_id}', (center_x, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return img

def create_tracked_pose_annotation(pose_results, video_path, n_person=2):
    """
    트래킹 결과를 기반으로 시간적 연속성을 보장하는 annotation을 생성합니다.
    """
    if not pose_results:
        return None

    # 1. 모든 프레임에서 유효한 track_id 수집
    all_track_ids = set()
    for result in pose_results:
        if hasattr(result._pred_instances, 'track_ids'):
            for tid in result._pred_instances.track_ids:
                if tid >= 0:
                    all_track_ids.add(tid)
    
    if not all_track_ids:
        print("Warning: No valid tracks found. Cannot create annotation.")
        return None

    # 2. 가장 오래 지속된 track_id를 n_person 만큼 선택
    track_lengths = {tid: 0 for tid in all_track_ids}
    for result in pose_results:
        if hasattr(result._pred_instances, 'track_ids'):
            for tid in result._pred_instances.track_ids:
                if tid in track_lengths:
                    track_lengths[tid] += 1
    
    # 길이를 기준으로 정렬하여 상위 n_person개의 ID 선택
    sorted_tracks = sorted(track_lengths.items(), key=lambda item: item[1], reverse=True)
    final_track_ids = [item[0] for item in sorted_tracks[:n_person]]
    
    # track_id를 배열 인덱스(0, 1, ...)로 매핑
    track_id_to_idx = {tid: i for i, tid in enumerate(final_track_ids)}
    
    num_persons = len(final_track_ids)
    num_frames = len(pose_results)
    # 첫 번째 유효한 결과에서 키포인트 수 가져오기
    num_keypoints = 17 # 기본값
    for res in pose_results:
        if hasattr(res._pred_instances, 'keypoints') and len(res._pred_instances.keypoints) > 0:
            num_keypoints = res._pred_instances.keypoints.shape[1]
            break

    # 3. 최종 배열 초기화
    keypoints = np.zeros((num_persons, num_frames, num_keypoints, 2), dtype=np.float32)
    scores = np.zeros((num_persons, num_frames, num_keypoints), dtype=np.float32)

    # 4. track_id를 기준으로 데이터 채우기
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
                keypoints[person_idx, f_idx] = instance_keypoints[p_idx]
                scores[person_idx, f_idx] = instance_scores[p_idx]

    # 5. 빈 프레임 보간 (간단한 이전 프레임 복사 방식)
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
        'img_shape': (pose_results[0].height, pose_results[0].width),
        'original_shape': (pose_results[0].height, pose_results[0].width),
        'total_frames': num_frames,
        'label': 0 # 필요시 레이블 설정
    }

def parse_args():
    # parse_args 함수는 변경 없이 그대로 사용
    parser = ArgumentParser(description='RTMO 포즈 추정 및 트래킹')
    parser.add_argument('config', help='RTMO config file')
    parser.add_argument('checkpoint', help='RTMO checkpoint file')
    parser.add_argument('--input', type=str, \
            default='/aivanas/raw/surveillance/action/violence/action_recognition/data/UBI_FIGHTS/videos/fight/F_4_0_0_0_0.mp4', \
            help='Video file path')
    parser.add_argument('--output-root', type=str, \
            default='/workspace/output', help='Output directory')
    parser.add_argument('--save-annotation', action='store_true', default=False, help='Save pose annotation file')
    parser.add_argument('--n-person', type=int, default=2, help='Number of main actors to track for annotation')
    parser.add_argument('--show', action='store_true', default=True, help='Show visualization')
    parser.add_argument('--device', default='cuda:0', help='Device for inference')
    parser.add_argument('--kpt-thr', type=float, default=0.9, help='Keypoint score threshold')
    parser.add_argument('--radius', type=int, default=2, help='Keypoint radius')
    parser.add_argument('--thickness', type=int, default=1, help='Link thickness')
    parser.add_argument('--show-interval', type=float, default=0, help='Sleep seconds per frame')
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.output_root:
        mmengine.mkdir_or_exist(args.output_root)
    
    model = init_model(args.config, args.checkpoint, device=args.device)
    
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.set_dataset_meta(model.dataset_meta, skeleton_style='mmpose')
    
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise ValueError(f'Cannot open video file: {args.input}')
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pose_results = []
    bbox_list = []

    # tqdm을 수동 update 방식으로 사용, 실시간 FPS 표시
    from tqdm import tqdm
    os.system('clear')
    import time as _time
    start_time = _time.time()
    pbar = tqdm(total=total_frames, desc="Step 1: Running pose estimation")
    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        batch_results = inference_bottomup(model, frame)
        result = batch_results[0]
        pose_results.append(result)

        pred_instances = result._pred_instances
        if hasattr(pred_instances, 'bboxes') and hasattr(pred_instances, 'bbox_scores'):
            bbox_scores = pred_instances.bbox_scores.reshape(-1, 1)
            frame_bboxes = np.concatenate([pred_instances.bboxes, bbox_scores], axis=1)
        else:
            frame_bboxes = np.empty((0, 5))
        bbox_list.append(frame_bboxes)

        frame_count += 1
        elapsed = _time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        pbar.set_postfix({'FPS': f'{fps:.2f}'})
        pbar.update(1)

    cap.release()
    pbar.close()
    elapsed = _time.time() - start_time
    fps = len(pose_results) / elapsed if elapsed > 0 else 0
    # ----------------------------------------------
    
    print("Step 2: Performing tracking...")
    tracklets = bbox2tracklet(bbox_list)
    tracklets = filter_tracklets(tracklets)
    
    frame_assignments = defaultdict(list)
    for tracklet_id, tracklet in tracklets.items():
        for frame_idx, bbox in tracklet:
            frame_assignments[frame_idx].append((tracklet_id, bbox))
            
    for idx, pose_result in enumerate(pose_results):
        pose_results[idx] = assign_track_ids(pose_result, frame_assignments, idx)
    
    print(f"Tracking completed. Found {len(tracklets)} tracks.")

    if args.save_annotation:
        print("Step 3: Creating and saving tracked pose annotation...")
        annotation = create_tracked_pose_annotation(pose_results, args.input, n_person=args.n_person)
        
        if annotation is not None:
            fname = os.path.splitext(os.path.basename(args.input))[0]
            annotation_path = os.path.join(args.output_root, f'{fname}_rtmo_pose.pkl')
            mmengine.dump(annotation, annotation_path)
            print(f'Pose annotation saved to: {annotation_path}')
        else:
            print("Warning: Could not create pose annotation.")

    # --- 4단계 수정: AttributeError 해결 ---
    if args.show:
        print("Step 4: Showing visualization...")
        cap = cv2.VideoCapture(args.input)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        wait_time = int(1000 / video_fps) if video_fps > 0 else 33  # 기본 30fps
        for idx, result in enumerate(pose_results):
            success, frame = cap.read()
            if not success:
                break
            # add_datasample을 사용하여 시각화
            visualizer.add_datasample(
                'result',
                frame,
                data_sample=result,
                draw_gt=False,
                draw_bbox=False,
                draw_heatmap=False,
                show_kpt_idx=False,
                show=False,
                wait_time=0,
                kpt_thr=args.kpt_thr
            )
            
            vis_img = draw_tracking_info(visualizer.get_image(),
                                         result.pred_instances,
                                         result.pred_instances.track_ids)
            
            cv2.imshow('RTMO Tracking', vis_img)
            if cv2.waitKey(wait_time) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
    # ----------------------------------------------
        
    print("Processing completed!")

if __name__ == '__main__':
    main()