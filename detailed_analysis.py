#!/usr/bin/env python3
import pickle
import numpy as np

def detailed_keypoint_analysis(pkl_file_path):
    """Perform detailed analysis of keypoint data structure."""
    
    with open(pkl_file_path, 'rb') as f:
        data = pickle.load(f)
    
    keypoints = data['keypoint']  # Shape: (61, 150, 17, 2)
    scores = data['keypoint_score']  # Shape: (61, 150, 17)
    
    print("=== DETAILED KEYPOINT ANALYSIS ===")
    print(f"Video: {data['frame_dir']}")
    print(f"Label: {data['label']} (1=Fight)")
    print(f"Total frames in video: {data['total_frames']}")
    print(f"Sequence length used: {keypoints.shape[0]} frames")
    print(f"Image resolution: {data['img_shape']}")
    print()
    
    # Analyze person validity across frames
    print("=== PERSON TRACKING ANALYSIS ===")
    
    # Check which persons have valid keypoints (confidence > threshold)
    confidence_threshold = 0.1
    valid_persons_per_frame = []
    
    for frame_idx in range(min(10, keypoints.shape[0])):
        frame_scores = scores[frame_idx]  # Shape: (150, 17)
        
        # A person is valid if they have at least some keypoints with good confidence
        min_keypoints_required = 5  # At least 5 keypoints should be confident
        valid_keypoints_per_person = np.sum(frame_scores > confidence_threshold, axis=1)
        valid_persons = valid_keypoints_per_person >= min_keypoints_required
        num_valid_persons = np.sum(valid_persons)
        valid_persons_per_frame.append(num_valid_persons)
        
        print(f"Frame {frame_idx:2d}: {num_valid_persons:2d} valid persons (>{min_keypoints_required} confident keypoints)")
        
        # Show details for first few valid persons
        valid_indices = np.where(valid_persons)[0][:3]  # First 3 valid persons
        for person_idx in valid_indices:
            avg_confidence = np.mean(frame_scores[person_idx])
            confident_kpts = np.sum(frame_scores[person_idx] > confidence_threshold)
            print(f"  Person {person_idx:3d}: {confident_kpts:2d}/17 confident keypoints, avg_conf={avg_confidence:.3f}")
    
    print()
    
    # Analyze keypoint structure (which keypoints are which)
    print("=== KEYPOINT STRUCTURE ANALYSIS ===")
    print("Based on MMPose COCO-17 format:")
    coco_keypoints = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    # Analyze confidence distribution per keypoint
    frame_0_scores = scores[0]  # First frame
    valid_person_mask = np.sum(frame_0_scores > confidence_threshold, axis=1) >= 5
    valid_person_scores = frame_0_scores[valid_person_mask]
    
    if len(valid_person_scores) > 0:
        avg_confidence_per_keypoint = np.mean(valid_person_scores, axis=0)
        
        print("Average confidence per keypoint (first frame, valid persons only):")
        for i, (kpt_name, conf) in enumerate(zip(coco_keypoints, avg_confidence_per_keypoint)):
            print(f"  {i:2d}. {kpt_name:15s}: {conf:.3f}")
    
    print()
    
    # Check for temporal consistency (potential tracking)
    print("=== TEMPORAL CONSISTENCY ANALYSIS ===")
    
    # Look at first few persons across first few frames
    for person_idx in range(min(3, keypoints.shape[1])):
        print(f"Person {person_idx} across frames:")
        
        # Check if this person appears consistently
        person_validity = []
        for frame_idx in range(min(10, keypoints.shape[0])):
            frame_scores = scores[frame_idx, person_idx]
            confident_kpts = np.sum(frame_scores > confidence_threshold)
            is_valid = confident_kpts >= 5
            person_validity.append(is_valid)
            
            if is_valid:
                # Show center position (average of confident keypoints)
                confident_mask = frame_scores > confidence_threshold
                if np.any(confident_mask):
                    confident_positions = keypoints[frame_idx, person_idx][confident_mask]
                    center_x = np.mean(confident_positions[:, 0])
                    center_y = np.mean(confident_positions[:, 1])
                    print(f"  Frame {frame_idx:2d}: Valid ({confident_kpts:2d} kpts) - Center: ({center_x:.1f}, {center_y:.1f})")
                else:
                    print(f"  Frame {frame_idx:2d}: Valid but no confident positions")
            else:
                print(f"  Frame {frame_idx:2d}: Invalid ({confident_kpts:2d} kpts)")
        
        consistency = np.sum(person_validity) / len(person_validity)
        print(f"  Temporal consistency: {consistency:.2f} ({np.sum(person_validity)}/{len(person_validity)} frames)")
        print()
    
    print("=== DATA FORMAT SUMMARY ===")
    print("Structure:")
    print("  - keypoint: (frames, max_persons, 17_keypoints, 2_coords) = (61, 150, 17, 2)")
    print("  - keypoint_score: (frames, max_persons, 17_keypoints) = (61, 150, 17)")
    print("  - Coordinates: (x, y) pixel positions")
    print("  - Scores: confidence values [0-1]")
    print("  - Person slots: Fixed 150 slots, filled in detection order")
    print("  - Tracking: Implicit through consistent slot assignment")
    print("  - Keypoint format: COCO-17 body keypoints")

if __name__ == "__main__":
    pkl_file = "/home/gaonpf/hsnam/mmlabs/mmaction2/data/rtmo_0.3_0.65_0.1/train/Fight/0H2s9UJcNJ0_0.pkl"
    detailed_keypoint_analysis(pkl_file)