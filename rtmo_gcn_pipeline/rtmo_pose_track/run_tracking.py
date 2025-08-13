import cv2
import numpy as np
import os
import time
import sys
from mmpose.apis import MMPoseInferencer
from mmpose.visualization import PoseLocalVisualizer
from mmpose.structures import PoseDataSample
from mmengine.structures import InstanceData

from core.tracker import ByteTracker

def main():
    # --- Configuration ---
    video_path = '/workspace/rtmo_gcn_pipeline/rtmo_pose_track/raw_videos/Fight/cam04_06.mp4'
    pose_config = '/workspace/mmpose/configs/body_2d_keypoint/rtmo/body7/rtmo-m_16xb16-600e_body7-640x640.py'
    pose_checkpoint = '/workspace/mmpose/checkpoints/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.pth'
    output_dir = '/workspace/rtmo_gcn_pipeline/rtmo_pose_track/output/'
    
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, 'tracked_cam04_06.mp4')

    # --- Initialization ---
    print("Initializing MMPoseInferencer...")
    pose_inferencer = MMPoseInferencer(
        pose2d=pose_config,
        pose2d_weights=pose_checkpoint,
        device='cuda:0'
    )

    print("Initializing ByteTracker...")
    tracker = ByteTracker(high_thresh=0.5, low_thresh=0.2, max_disappeared=50, min_hits=3)

    visualizer = PoseLocalVisualizer()

    # --- Video Processing ---
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    total_time = 0
    
    print("Starting video processing...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        start_time = time.time()

        result_generator = pose_inferencer(frame, show=False)
        pose_result_dict = next(result_generator)
        predictions = pose_result_dict['predictions'][0]

        # Convert predictions to ByteTracker format
        detections = []
        for pred in predictions:
            bbox = pred['bbox'][0]
            score = pred['bbox_score']
            detections.append(bbox + [score])
        detections = np.array(detections)

        # Update tracker
        active_tracks = tracker.update(detections)

        # Create a PoseDataSample for visualization
        data_sample = PoseDataSample()
        pred_instances = InstanceData()
        
        # Prepare data for InstanceData
        bboxes = np.array([track.to_bbox() for track in active_tracks])
        track_ids = np.array([track.track_id for track in active_tracks])
        
        # Match keypoints to tracks
        keypoints_list = []
        keypoints_scores_list = []
        if len(predictions) > 0 and len(active_tracks) > 0:
            for track in active_tracks:
                best_match_idx = -1
                max_iou = 0.3 # IoU threshold
                track_bbox = track.to_bbox()
                for i, pred in enumerate(predictions):
                    pred_bbox = pred['bbox'][0]
                    iou = ByteTracker._compute_iou(None, track_bbox, pred_bbox)
                    if iou > max_iou:
                        max_iou = iou
                        best_match_idx = i
                
                if best_match_idx != -1:
                    keypoints_list.append(predictions[best_match_idx]['keypoints'])
                    keypoints_scores_list.append(predictions[best_match_idx]['keypoint_scores'])
                else: # if no match found, append empty kpts
                    keypoints_list.append(np.zeros((17, 2)))
                    keypoints_scores_list.append(np.zeros(17))

        pred_instances.bboxes = bboxes
        pred_instances.track_ids = track_ids
        if keypoints_list:
            pred_instances.keypoints = np.array(keypoints_list)
            pred_instances.keypoint_scores = np.array(keypoints_scores_list)

        data_sample.pred_instances = pred_instances

        end_time = time.time()
        total_time += (end_time - start_time)

        # Visualization
        visualizer.add_datasample(
            'result',
            frame,
            data_sample=data_sample,
            draw_gt=False,
            draw_heatmap=False,
            draw_bbox=True,
            show_kpt_idx=False,
            skeleton_style='mmpose',
            show=False,
            out_file=None
        )
        
        vis_frame = visualizer.get_image()

        # Draw tracking IDs on the joints
        if hasattr(data_sample.pred_instances, 'track_ids') and hasattr(data_sample.pred_instances, 'keypoints'):
            for i, track_id in enumerate(data_sample.pred_instances.track_ids):
                if track_id == -1:
                    continue

                # Find the highest visible keypoint to place the ID
                person_keypoints = data_sample.pred_instances.keypoints[i]
                person_scores = data_sample.pred_instances.keypoint_scores[i]
                
                highest_point = None
                min_y = float('inf')

                for j, (x, y) in enumerate(person_keypoints):
                    if person_scores[j] > 0.3 and y < min_y:
                        min_y = y
                        highest_point = (int(x), int(y))
                
                # If a keypoint is found, draw the ID above it
                if highest_point:
                    text_pos = (highest_point[0], highest_point[1] - 15)
                    cv2.putText(vis_frame, f"ID: {track_id}", text_pos, 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                # Fallback to the bounding box top if no suitable keypoint is found
                else:
                    bbox = data_sample.pred_instances.bboxes[i]
                    cv2.putText(vis_frame, f"ID: {track_id}", (int(bbox[0]), int(bbox[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        video_writer.write(vis_frame)

        # Show the result in a window for debugging
        cv2.imshow('RTMO-GCN Tracking', vis_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        if frame_count % 10 == 0:
            print(f"Processed frame {frame_count}...")

    # --- Cleanup and Report ---
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    
    print("\n--- Tracking Finished ---")
    print(f"Output video saved to: {output_video_path}")
    
    avg_fps = frame_count / total_time if total_time > 0 else 0
    print("\n--- Performance Metrics ---")
    print(f"Total frames processed: {frame_count}")
    print(f"Total tracks created: {tracker.next_id}")
    print(f"Average processing time: {total_time / frame_count:.4f} seconds/frame")
    print(f"Average FPS: {avg_fps:.2f}")
    print("-------------------------\n")

if __name__ == '__main__':
    # A helper function for IoU calculation inside ByteTracker
    def _compute_iou(self, box1, box2):
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
    ByteTracker._compute_iou = _compute_iou

    main()