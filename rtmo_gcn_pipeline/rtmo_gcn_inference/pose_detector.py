# pose_detector.py - 포즈 추정 전용 모듈
import numpy as np
import cv2
import torch
import mmpose.datasets.transforms
import mmaction.datasets.transforms
from mmpose.apis import init_model, inference_bottomup
from mmpose.structures import PoseDataSample

class PoseDetector:
    def __init__(self):
        self.model = None
        self.device = 'cuda:0'
        self.nms_thr = 0.65
        self.score_thr = 0.3
        
    def init_model(self, config_path, checkpoint_path, device='cuda:0', nms_thr=0.65, score_thr=0.1):
        """포즈 모델 초기화"""
        try:
            self.device = device
            self.nms_thr = nms_thr
            self.score_thr = score_thr
            
            # MMPose registry를 먼저 import
            import mmpose.datasets.transforms  # 이 라인 추가
            
            self.model = init_model(
                config_path,
                checkpoint_path,
                device=device
            )
            
            # 모델 설정 업데이트
            if hasattr(self.model, 'cfg') and hasattr(self.model.cfg, 'model'):
                if 'test_cfg' in self.model.cfg.model:
                    self.model.cfg.model['test_cfg']['nms_thr'] = self.nms_thr
                    self.model.cfg.model['test_cfg']['score_thr'] = self.score_thr
            
            if hasattr(self.model, 'test_cfg'):
                self.model.test_cfg.nms_thr = self.nms_thr
                self.model.test_cfg.score_thr = self.score_thr
            
            return True, "Pose model initialized successfully!"
        except Exception as e:
            return False, f"Error initializing pose model: {e}"
    
    def extract_pose_data(self, result):
        """포즈 추정 결과에서 데이터 추출"""
        try:
            if isinstance(result, list):
                if len(result) == 0:
                    return None
                result = result[0]
            
            if not isinstance(result, PoseDataSample):
                return None
            
            if not hasattr(result, 'pred_instances'):
                return None
            
            pred_instances = result.pred_instances
            if len(pred_instances) == 0:
                return None
            
            return pred_instances
        
        except Exception as e:
            print(f"Error extracting pose data: {e}")
            return None
    
    def detect_pose(self, image):
        """포즈 추정 수행"""
        if self.model is None:
            return None, None, None
        
        try:
            # 이미지 배열로 직접 추론
            result = inference_bottomup(self.model, image)
            pred_instances = self.extract_pose_data(result)
            
            if pred_instances is not None:
                keypoints = []
                scores = []
                
                for instance in pred_instances:
                    if hasattr(instance, 'keypoints') and hasattr(instance, 'keypoint_scores'):
                        kpts = instance.keypoints[0] if len(instance.keypoints.shape) > 2 else instance.keypoints
                        scrs = instance.keypoint_scores[0] if len(instance.keypoint_scores.shape) > 1 else instance.keypoint_scores
                        keypoints.append(kpts)
                        scores.append(scrs)
                
                if keypoints:
                    keypoints = np.array(keypoints)
                    scores = np.array(scores)
                    return keypoints, scores, image
            
            return None, None, image
            
        except Exception as e:
            print(f"Error in pose detection: {e}")
            return None, None, image
    
    def draw_keypoints(self, image, keypoints, keypoint_scores, score_threshold=0.3):
        """키포인트 그리기"""
        if keypoints is None or keypoint_scores is None:
            return image
            
        skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
            [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
            [2, 4], [3, 5], [4, 6], [5, 7]
        ]
        
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), 
                  (0, 255, 255), (128, 0, 128), (255, 165, 0), (255, 192, 203), (255, 20, 147)]
        
        for person_idx in range(keypoints.shape[0]):
            person_keypoints = keypoints[person_idx]
            person_scores = keypoint_scores[person_idx]
            color = colors[person_idx % len(colors)]
            
            # 키포인트 그리기
            for i, (x, y) in enumerate(person_keypoints):
                if person_scores[i] > score_threshold:
                    cv2.circle(image, (int(x), int(y)), 3, color, -1)
            
            # 스켈레톤 그리기
            for connection in skeleton:
                pt1_idx, pt2_idx = connection[0] - 1, connection[1] - 1
                if (pt1_idx < len(person_keypoints) and pt2_idx < len(person_keypoints) and
                    person_scores[pt1_idx] > score_threshold and person_scores[pt2_idx] > score_threshold):
                    pt1 = tuple(map(int, person_keypoints[pt1_idx]))
                    pt2 = tuple(map(int, person_keypoints[pt2_idx]))
                    cv2.line(image, pt1, pt2, color, 2)
        
        return image