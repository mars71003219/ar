#!/usr/bin/env python3
import pickle
import numpy as np
import sys

def analyze_pkl_structure(pkl_file_path):
    """Analyze the structure of a PKL file used for STGCN++ training."""
    
    print("=== PKL FILE STRUCTURE ANALYSIS ===")
    print(f"File: {pkl_file_path}")
    print()
    
    try:
        with open(pkl_file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Type of loaded data: {type(data)}")
        print(f"Data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dictionary'}")
        print()
        
        if isinstance(data, dict):
            for key, value in data.items():
                print(f"Key: '{key}'")
                print(f"  Type: {type(value)}")
                
                if isinstance(value, np.ndarray):
                    print(f"  Shape: {value.shape}")
                    print(f"  Dtype: {value.dtype}")
                    print(f"  Size: {value.size}")
                    
                    # Special handling for keypoint data
                    if 'keypoint' in key.lower() and len(value.shape) >= 3:
                        print(f"  Likely format: [frames, persons, keypoints, coordinates]")
                        if len(value.shape) == 4:
                            frames, persons, keypoints, coords = value.shape
                            print(f"  Frames: {frames}, Persons: {persons}, Keypoints: {keypoints}, Coordinates: {coords}")
                        elif len(value.shape) == 3:
                            dim1, dim2, dim3 = value.shape
                            print(f"  Dimensions: {dim1} x {dim2} x {dim3}")
                    
                    # Show a small sample of the data
                    if value.size < 100:
                        print(f"  Content: {value}")
                    else:
                        print(f"  Content sample (first 10 values): {value.flat[:10]}")
                        
                elif isinstance(value, (list, tuple)):
                    print(f"  Length: {len(value)}")
                    if len(value) > 0:
                        print(f"  First element type: {type(value[0])}")
                        if isinstance(value[0], np.ndarray):
                            print(f"  First element shape: {value[0].shape}")
                        if len(value) < 10:
                            print(f"  Content: {value}")
                        else:
                            print(f"  First few elements: {value[:3]}")
                            
                else:
                    print(f"  Value: {value}")
                print()
                
        # Additional analysis for keypoint data
        if isinstance(data, dict) and 'keypoint' in data:
            keypoint_data = data['keypoint']
            print("=== KEYPOINT DATA DETAILED ANALYSIS ===")
            print(f"Keypoint array shape: {keypoint_data.shape}")
            
            if len(keypoint_data.shape) == 4:
                frames, persons, keypoints, coords = keypoint_data.shape
                print(f"Temporal sequence length: {frames} frames")
                print(f"Maximum persons tracked: {persons}")
                print(f"Number of keypoints per person: {keypoints}")
                print(f"Coordinates per keypoint: {coords} (likely x, y, confidence)")
                
                # Check for valid keypoints (non-zero)
                valid_frames = []
                for frame_idx in range(min(5, frames)):  # Check first 5 frames
                    frame_data = keypoint_data[frame_idx]
                    valid_persons = np.any(frame_data.reshape(persons, -1) != 0, axis=1)
                    num_valid_persons = np.sum(valid_persons)
                    valid_frames.append(num_valid_persons)
                    print(f"Frame {frame_idx}: {num_valid_persons} valid persons")
                
                # Sample keypoint values
                print(f"\nSample keypoint values from frame 0, person 0:")
                sample_keypoints = keypoint_data[0, 0, :5, :]  # First 5 keypoints
                print(sample_keypoints)
                
            print()
            
        # Check for tracking information
        if isinstance(data, dict):
            tracking_keys = [k for k in data.keys() if 'track' in k.lower() or 'id' in k.lower()]
            if tracking_keys:
                print("=== TRACKING INFORMATION ===")
                for key in tracking_keys:
                    print(f"Found tracking key: '{key}'")
                    print(f"  Value: {data[key]}")
                print()
            else:
                print("=== NO EXPLICIT TRACKING KEYS FOUND ===")
                print("Tracking may be implicit in person ordering across frames")
                print()
                
    except Exception as e:
        print(f"Error loading PKL file: {e}")
        return

if __name__ == "__main__":
    pkl_file = "/home/gaonpf/hsnam/mmlabs/mmaction2/data/rtmo_0.3_0.65_0.1/train/Fight/0H2s9UJcNJ0_0.pkl"
    analyze_pkl_structure(pkl_file)