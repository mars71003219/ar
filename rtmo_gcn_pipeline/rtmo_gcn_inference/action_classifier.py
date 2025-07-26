# action_classifier.py - 액션 분류 전용 모듈 (수정된 버전)
import numpy as np
import os.path as osp
from mmaction.apis import init_recognizer, inference_skeleton
from mmaction.utils import frame_extract

class ActionClassifier:
    def __init__(self):
        self.model = None
        self.device = 'cuda:0'
        self.sequence_length = 30
        self.confidence_threshold = 0.5
        
    def init_model(self, config_path, checkpoint_path, device='cuda:0', sequence_length=30, confidence_threshold=0.5):
        """GCN 모델 초기화"""
        try:
            self.device = device
            self.sequence_length = sequence_length
            self.confidence_threshold = confidence_threshold
            
            # 빈 경로가 제공된 경우 모델 로드하지 않음
            if not config_path or not checkpoint_path:
                self.model = None
                return True, "GCN model not initialized (empty paths provided). Using default predictions."
            
            if not osp.exists(config_path):
                return False, f"Config file not found: {config_path}"
            
            if not osp.exists(checkpoint_path):
                return False, f"Checkpoint file not found: {checkpoint_path}"
            
            # mmaction2 API를 사용하여 모델 초기화
            self.model = init_recognizer(
                config=config_path,
                checkpoint=checkpoint_path,
                device=device
            )
            
            return True, "GCN model initialized successfully with mmaction2 API!"
        
        except Exception as e:
            return False, f"Error initializing GCN model: {e}"
    
    def normalize_keypoints_per_frame(self, keypoints_sequence, scores_sequence):
        """각 프레임의 키포인트를 정규화하여 일관된 형태로 만듦"""
        normalized_keypoints = []
        normalized_scores = []
        
        for frame_keypoints, frame_scores in zip(keypoints_sequence, scores_sequence):
            # 프레임별 키포인트 처리
            if len(frame_keypoints) == 0:
                # 검출된 사람이 없는 경우 더미 데이터 생성
                dummy_keypoints = np.zeros((17, 2))
                dummy_scores = np.zeros(17)
                normalized_keypoints.append(dummy_keypoints)
                normalized_scores.append(dummy_scores)
            else:
                # 첫 번째 사람의 키포인트만 사용 (가장 신뢰도가 높은 것으로 가정)
                if len(frame_keypoints.shape) == 3:  # (num_person, 17, 2)
                    person_keypoints = frame_keypoints[0]  # (17, 2)
                    person_scores = frame_scores[0]       # (17,)
                else:  # (17, 2)
                    person_keypoints = frame_keypoints
                    person_scores = frame_scores
                
                # 키포인트 유효성 검사
                if person_keypoints.shape[0] != 17:
                    # 키포인트 수가 맞지 않으면 더미 데이터 사용
                    person_keypoints = np.zeros((17, 2))
                    person_scores = np.zeros(17)
                
                normalized_keypoints.append(person_keypoints)
                normalized_scores.append(person_scores)
        
        return normalized_keypoints, normalized_scores
    
    def preprocess_keypoints(self, keypoints_sequence, scores_sequence):
        """GCN 입력을 위한 키포인트 전처리"""
        try:
            # 키포인트 정규화
            normalized_keypoints, normalized_scores = self.normalize_keypoints_per_frame(
                keypoints_sequence, scores_sequence
            )
            
            # 시퀀스 길이 조정
            if len(normalized_keypoints) < self.sequence_length:
                # 시퀀스가 짧으면 패딩
                needed = self.sequence_length - len(normalized_keypoints)
                if len(normalized_keypoints) > 0:
                    # 마지막 프레임 반복
                    last_keypoints = normalized_keypoints[-1]
                    last_scores = normalized_scores[-1]
                    for _ in range(needed):
                        normalized_keypoints.append(last_keypoints.copy())
                        normalized_scores.append(last_scores.copy())
                else:
                    # 빈 시퀀스면 0으로 채움
                    dummy_keypoints = np.zeros((17, 2))
                    dummy_scores = np.zeros(17)
                    for _ in range(self.sequence_length):
                        normalized_keypoints.append(dummy_keypoints.copy())
                        normalized_scores.append(dummy_scores.copy())
            
            # 시퀀스 길이 맞추기 (최신 프레임들만 사용)
            normalized_keypoints = normalized_keypoints[-self.sequence_length:]
            normalized_scores = normalized_scores[-self.sequence_length:]
            
            # 배열로 변환 (이제 모든 프레임이 동일한 형태를 가짐)
            keypoints_array = np.array(normalized_keypoints)  # (seq_len, 17, 2)
            scores_array = np.array(normalized_scores)        # (seq_len, 17)
            
            # 신뢰도가 낮은 키포인트 마스킹
            mask = scores_array > 0.3  # pose_score_thr
            keypoints_array[~mask] = 0
            
            return keypoints_array, scores_array
        
        except Exception as e:
            print(f"Error in preprocess_keypoints: {e}")
            # 에러 발생 시 더미 데이터 반환
            dummy_keypoints = np.zeros((self.sequence_length, 17, 2))
            dummy_scores = np.zeros((self.sequence_length, 17))
            return dummy_keypoints, dummy_scores
    
    def predict(self, keypoints_sequence, scores_sequence):
        """GCN 모델로 예측 수행"""
        try:
            if self.model is None:
                return 0, 0.5  # 기본값
            
            # 키포인트 전처리
            keypoints, scores = self.preprocess_keypoints(keypoints_sequence, scores_sequence)
            
            # mmaction2 형태로 변환
            pose_results = []
            for i in range(keypoints.shape[0]):
                frame_keypoints = keypoints[i:i+1]  # (1, 17, 2)
                frame_scores = scores[i:i+1]       # (1, 17)
                
                frame_result = {
                    'keypoints': frame_keypoints,
                    'keypoint_scores': frame_scores
                }
                pose_results.append(frame_result)
            
            # 이미지 크기 (더미 값)
            img_shape = (480, 640)  # (height, width)
            
            # inference_skeleton 함수 사용
            result = inference_skeleton(
                model=self.model,
                pose_results=pose_results,
                img_shape=img_shape
            )
            
            # 결과 처리
            if hasattr(result, 'pred_score'):
                pred_scores = result.pred_score.cpu().numpy()
                confidence = float(np.max(pred_scores))
                prediction = int(np.argmax(pred_scores))
            else:
                # 대체 방법
                confidence = 0.5
                prediction = 0
            
            return prediction, confidence
        
        except Exception as e:
            print(f"Error in GCN prediction: {e}")
            return 0, 0.5
    
    def extract_frames(self, vid_path, out_dir):
        """비디오에서 프레임 추출"""
        try:
            return frame_extract(vid_path, out_dir=out_dir)
        except Exception as e:
            print(f"Error extracting frames: {e}")
            return None