#!/usr/bin/env python3
"""
RTMO ONNX 포즈 추정기 구현

rtmlib의 RTMO 구현을 기반으로 recognizer 프레임워크에 맞게 최적화된 버전입니다.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
import logging

try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False
    logging.warning("ONNXRuntime not available")

try:
    from pose_estimation.base import BasePoseEstimator
    from utils.data_structure import PersonPose, FramePoses, PoseEstimationConfig
except ImportError:
    from ..base import BasePoseEstimator
    from ...utils.data_structure import PersonPose, FramePoses, PoseEstimationConfig


class RTMOONNXEstimator(BasePoseEstimator):
    """RTMO ONNX 포즈 추정기 구현"""
    
    def __init__(self, config: PoseEstimationConfig):
        """
        Args:
            config: RTMO ONNX 포즈 추정 설정
        """
        super().__init__(config)
        
        if not ONNXRUNTIME_AVAILABLE:
            raise RuntimeError("ONNXRuntime is required for ONNX pose estimation")
        
        # ONNX 특화 설정
        self.onnx_model_path = config.model_path  # ONNX 모델 경로
        self.model_input_size = getattr(config, 'model_input_size', (640, 640))
        
        # model_input_size가 None이거나 잘못된 경우 기본값 설정
        if self.model_input_size is None or len(self.model_input_size) != 2:
            self.model_input_size = (640, 640)
            logging.warning(f"Invalid model_input_size, using default: {self.model_input_size}")
        self.mean = getattr(config, 'mean', None)
        self.std = getattr(config, 'std', None)
        self.backend = getattr(config, 'backend', 'onnxruntime')
        self.to_openpose = getattr(config, 'to_openpose', False)
        
        # ONNX Runtime 세션
        self.session = None
        
        # 통계
        self.stats = {
            'total_frames_processed': 0,
            'total_persons_detected': 0,
            'avg_persons_per_frame': 0.0,
            'processing_times': []
        }
        
        # 자동 초기화
        self.initialize_model()
    
    def initialize_model(self) -> bool:
        """ONNX 모델 초기화"""
        try:
            if self.is_initialized:
                return True
                
            if not os.path.exists(self.onnx_model_path):
                raise FileNotFoundError(f"ONNX model not found: {self.onnx_model_path}")
            
            # CUDA 12 환경변수 설정
            os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-12.1/targets/x86_64-linux/lib:/usr/local/cuda-12.1/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')
            os.environ['CUDA_PATH'] = '/usr/local/cuda-12.1'
            
            # ONNX Runtime 설정 (CUDA 실패시 CPU로 폴백)
            providers = self._get_providers()
            
            try:
                # 최적화된 세션 옵션 설정
                session_options = ort.SessionOptions()
                session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                session_options.enable_cpu_mem_arena = True
                session_options.enable_mem_pattern = True
                session_options.enable_mem_reuse = True
                session_options.log_severity_level = 3  # 경고 메시지 줄이기
                
                self.session = ort.InferenceSession(
                    path_or_bytes=self.onnx_model_path,
                    providers=providers,
                    sess_options=session_options
                )
                
                # 실제 사용 중인 제공자 확인
                actual_providers = self.session.get_providers()
                if self.device.startswith('cuda') and 'CUDAExecutionProvider' not in actual_providers:
                    logging.warning(f"CUDA 제공자를 요청했지만 사용할 수 없습니다. CPU로 실행됩니다.")
                    logging.warning(f"실제 사용 제공자: {actual_providers}")
                
            except Exception as e:
                logging.warning(f"지정된 제공자로 세션 생성 실패: {e}")
                logging.info("CPU 제공자로 재시도합니다.")
                
                # CPU 제공자로 다시 시도 (최적화된 옵션 사용)
                self.session = ort.InferenceSession(
                    path_or_bytes=self.onnx_model_path,
                    providers=['CPUExecutionProvider'],
                    sess_options=session_options
                )
            
            self.is_initialized = True
            print(f"RTMO ONNX model initialized successfully with {providers}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize RTMO ONNX model: {str(e)}")
            self.is_initialized = False
            return False
    
    def _get_providers(self) -> List[str]:
        """사용 가능한 실행 공급자 반환"""
        available_providers = ort.get_available_providers()
        
        # 디바이스에 따라 우선순위 설정
        if self.device.startswith('cuda') and 'CUDAExecutionProvider' in available_providers:
            return self._get_cuda_providers()
        elif self.device == 'cpu':
            return ['CPUExecutionProvider']
        else:
            # 기본값: CUDA 사용 가능하면 CUDA, 아니면 CPU
            if 'CUDAExecutionProvider' in available_providers:
                return self._get_cuda_providers()
            else:
                return ['CPUExecutionProvider']
    
    def _get_cuda_providers(self) -> List[str]:
        """CUDA 제공자 설정 반환"""
        # CUDA 디바이스 ID 추출
        device_id = 0
        if self.device.startswith('cuda:'):
            try:
                device_id = int(self.device.split(':')[1])
            except (IndexError, ValueError):
                device_id = 0
        
        # CUDA 제공자 옵션 설정 (최적화 - 13ms 달성 설정)
        cuda_provider_options = {
            'device_id': device_id,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': 20 * 1024 * 1024 * 1024,  # 20GB
            'cudnn_conv_algo_search': 'HEURISTIC',  # EXHAUSTIVE -> HEURISTIC (더 빠름)
            'do_copy_in_default_stream': False,  # False로 변경 (성능 향상)
        }
        
        # 제공자와 옵션을 튜플로 반환
        return [
            ('CUDAExecutionProvider', cuda_provider_options),
            'CPUExecutionProvider'
        ]
    
    def extract_poses(self, frame: np.ndarray, frame_idx: int = 0) -> List[PersonPose]:
        """단일 프레임에서 포즈 추출"""
        if not self.ensure_initialized():
            raise RuntimeError("Failed to initialize RTMO ONNX model")
        
        if not self.validate_frame(frame):
            return []
        
        try:
            import time
            start_time = time.time()
            
            # 프레임 확인
            if frame is None:
                logging.error(f"Frame {frame_idx} is None")
                return []
            
            # ONNX 추론
            keypoints, scores = self._inference_onnx(frame)
            
            # PersonPose 객체 생성
            persons = self._convert_to_persons(keypoints, scores, frame_idx)
            
            # 품질 필터링
            persons = self.filter_low_quality_poses(persons)
            
            # 통계 업데이트
            processing_time = time.time() - start_time
            self.stats['processing_times'].append(processing_time)
            self.stats['total_persons_detected'] += len(persons)
            
            return persons
            
        except Exception as e:
            import traceback
            logging.error(f"Error in ONNX pose extraction for frame {frame_idx}: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    def _inference_onnx(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ONNX 모델 추론 수행"""
        # 전처리
        preprocessed_img, ratio = self._preprocess(image)
        
        # 추론
        outputs = self._inference(preprocessed_img)
        if outputs is None:
            return np.empty((0, 17, 2)), np.empty((0, 17))
        
        # 후처리
        keypoints, scores = self._postprocess(outputs, ratio)
        
        # OpenPose 형식으로 변환 (옵션)
        if self.to_openpose:
            keypoints, scores = self._convert_coco_to_openpose(keypoints, scores)
        
        return keypoints, scores
    
    def _preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        """전처리: 이미지 리사이즈 및 패딩"""
        if img is None:
            raise ValueError("Input image is None")
        
        if len(img.shape) == 3:
            padded_img = np.ones(
                (self.model_input_size[0], self.model_input_size[1], 3),
                dtype=np.uint8) * 114
        else:
            padded_img = np.ones(self.model_input_size, dtype=np.uint8) * 114

        ratio = min(self.model_input_size[0] / img.shape[0],
                    self.model_input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * ratio), int(img.shape[0] * ratio)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_shape = (int(img.shape[0] * ratio), int(img.shape[1] * ratio))
        padded_img[:padded_shape[0], :padded_shape[1]] = resized_img

        # 정규화
        if self.mean is not None:
            mean = np.array(self.mean)
            std = np.array(self.std)
            padded_img = (padded_img - mean) / std

        return padded_img, ratio
    
    def _inference(self, img: np.ndarray) -> List[np.ndarray]:
        """ONNX 모델 추론 실행 (최적화)"""
        # 입력 형식 변환: (H, W, C) -> (1, C, H, W) - 한 번에 처리
        input_tensor = np.ascontiguousarray(
            img.transpose(2, 0, 1)[None, :, :, :], 
            dtype=np.float32
        )
        
        # 입/출력 이름 캐시 (첫 실행시만 계산)
        if not hasattr(self, '_input_name'):
            self._input_name = self.session.get_inputs()[0].name
            self._output_names = [output.name for output in self.session.get_outputs()]
        
        # 추론 실행
        try:
            outputs = self.session.run(self._output_names, {self._input_name: input_tensor})
            if outputs is None:
                logging.warning("ONNX session.run returned None")
                return None
            return outputs
        except Exception as e:
            logging.error(f"ONNX inference failed: {str(e)}")
            return None
    
    def _postprocess(self, outputs: List[np.ndarray], ratio: float) -> Tuple[np.ndarray, np.ndarray]:
        """후처리: NMS 적용 및 좌표 복원"""
        if outputs is None or len(outputs) != 2:
            logging.warning(f"Invalid ONNX outputs: {outputs}")
            return np.empty((0, 17, 2)), np.empty((0, 17))
        
        det_outputs, pose_outputs = outputs
        
        # 검출 결과 처리
        final_boxes = det_outputs[0, :, :4]
        final_scores = det_outputs[0, :, 4]
        final_boxes /= ratio
        
        # 포즈 결과 처리
        keypoints = pose_outputs[0, :, :, :2]
        scores = pose_outputs[0, :, :, 2]
        keypoints = keypoints / ratio
        
        # NMS 적용
        from .multiclass_nms import multiclass_nms
        dets, keep = multiclass_nms(
            final_boxes, 
            final_scores[:, np.newaxis],
            nms_thr=self.nms_threshold,
            score_thr=self.score_threshold
        )
        
        if keep is not None:
            keypoints = keypoints[keep]
            scores = scores[keep]
        else:
            keypoints = np.expand_dims(np.zeros_like(keypoints[0]), axis=0)
            scores = np.expand_dims(np.zeros_like(scores[0]), axis=0)
        
        return keypoints, scores
    
    def _convert_coco_to_openpose(self, keypoints: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """COCO 형식을 OpenPose 형식으로 변환"""
        # COCO (17 keypoints) to OpenPose (18 keypoints) 매핑
        coco_to_openpose = [0, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
        neck_keypoint_idx = (5 + 6) // 2  # 목 키포인트는 양쪽 어깨의 중점
        
        openpose_keypoints = np.zeros((keypoints.shape[0], 18, 2))
        openpose_scores = np.zeros((scores.shape[0], 18))
        
        for i, idx in enumerate(coco_to_openpose):
            openpose_keypoints[:, i] = keypoints[:, idx]
            openpose_scores[:, i] = scores[:, idx]
        
        # 목 키포인트 계산 (어깨 중점)
        left_shoulder = keypoints[:, 5]  # COCO left shoulder
        right_shoulder = keypoints[:, 6]  # COCO right shoulder
        neck_keypoint = (left_shoulder + right_shoulder) / 2
        neck_score = (scores[:, 5] + scores[:, 6]) / 2
        
        openpose_keypoints[:, 17] = neck_keypoint  # OpenPose neck
        openpose_scores[:, 17] = neck_score
        
        return openpose_keypoints, openpose_scores
    
    def _convert_to_persons(self, keypoints: np.ndarray, scores: np.ndarray, frame_idx: int) -> List[PersonPose]:
        """키포인트를 PersonPose 객체로 변환"""
        persons = []
        
        for person_idx in range(len(keypoints)):
            try:
                person_keypoints = keypoints[person_idx]  # (17, 2) or (18, 2)
                person_scores = scores[person_idx]  # (17,) or (18,)
                
                # (x, y, score) 형식으로 변환
                kpts_with_scores = np.zeros((len(person_keypoints), 3))
                kpts_with_scores[:, :2] = person_keypoints
                kpts_with_scores[:, 2] = person_scores
                
                # 바운딩 박스 계산
                valid_mask = person_scores > self.keypoint_threshold
                if not np.any(valid_mask):
                    continue
                    
                valid_keypoints = person_keypoints[valid_mask]
                if len(valid_keypoints) == 0:
                    continue
                
                x_coords = valid_keypoints[:, 0]
                y_coords = valid_keypoints[:, 1]
                
                x1, y1 = np.min(x_coords), np.min(y_coords)
                x2, y2 = np.max(x_coords), np.max(y_coords)
                
                # 바운딩 박스 확장
                padding = 0.1
                w, h = x2 - x1, y2 - y1
                x1 = max(0, x1 - w * padding)
                y1 = max(0, y1 - h * padding)
                x2 = x2 + w * padding
                y2 = y2 + h * padding
                
                bbox = [x1, y1, x2, y2]
                
                # 전체 신뢰도 점수
                overall_score = np.mean(person_scores[valid_mask])
                
                # PersonPose 생성
                person = PersonPose(
                    person_id=person_idx,
                    bbox=bbox,
                    keypoints=kpts_with_scores,
                    score=float(overall_score)
                )
                
                persons.append(person)
                
            except Exception as e:
                logging.warning(f"Error converting ONNX result {person_idx}: {str(e)}")
                continue
        
        return persons
    
    def process_frame(self, frame: np.ndarray, frame_idx: int = 0) -> FramePoses:
        """단일 프레임 처리 - 파이프라인 인터페이스용"""
        persons = self.extract_poses(frame, frame_idx)
        
        frame_poses = FramePoses(
            frame_idx=frame_idx,
            persons=persons,
            timestamp=frame_idx / 30.0,  # 기본 30 FPS 가정
            image_shape=(frame.shape[0], frame.shape[1]) if frame is not None else (0, 0)
        )
        
        return frame_poses
    
    def extract_video_poses(self, video_path: str) -> List[FramePoses]:
        """전체 비디오에서 포즈 추출"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        print(f"Extracting poses from: {video_path} using ONNX")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        # 비디오 정보
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frame_poses_list = []
        frame_idx = 0
        
        print(f"Video info: {total_frames} frames, {fps:.2f} FPS, {width}x{height}")
        
        with tqdm(total=total_frames, desc="Processing frames (ONNX)") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 포즈 추출
                persons = self.extract_poses(frame, frame_idx)
                
                # FramePoses 생성
                timestamp = frame_idx / fps if fps > 0 else frame_idx
                frame_poses = FramePoses(
                    frame_idx=frame_idx,
                    persons=persons,
                    timestamp=timestamp,
                    image_shape=(height, width)
                )
                
                frame_poses_list.append(frame_poses)
                
                frame_idx += 1
                pbar.update(1)
        
        cap.release()
        
        # 통계 업데이트
        self.stats['total_frames_processed'] += len(frame_poses_list)
        if len(frame_poses_list) > 0:
            total_persons = sum(len(fp.persons) for fp in frame_poses_list)
            self.stats['avg_persons_per_frame'] = total_persons / len(frame_poses_list)
        
        print(f"Extracted poses from {len(frame_poses_list)} frames using ONNX")
        print(f"Average persons per frame: {self.stats['avg_persons_per_frame']:.2f}")
        
        return frame_poses_list
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        info = {
            'model_name': 'RTMO-ONNX',
            'onnx_model_path': self.onnx_model_path,
            'model_input_size': self.model_input_size,
            'backend': self.backend,
            'device': self.device,
            'score_threshold': self.score_threshold,
            'nms_threshold': self.nms_threshold,
            'keypoint_threshold': self.keypoint_threshold,
            'max_detections': self.max_detections,
            'to_openpose': self.to_openpose,
            'is_initialized': self.is_initialized,
            'statistics': self.stats.copy()
        }
        
        if self.session:
            try:
                info['input_shape'] = [inp.shape for inp in self.session.get_inputs()]
                info['output_shape'] = [out.shape for out in self.session.get_outputs()]
                info['providers'] = self.session.get_providers()
            except:
                pass
        
        return info
    
    def cleanup(self):
        """리소스 정리"""
        super().cleanup()
        
        if self.session:
            del self.session
            self.session = None
        
        self.is_initialized = False
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """처리 통계 반환"""
        stats = self.stats.copy()
        
        if stats['processing_times']:
            stats['avg_processing_time'] = np.mean(stats['processing_times'])
            stats['std_processing_time'] = np.std(stats['processing_times'])
            stats['fps_estimate'] = 1.0 / np.mean(stats['processing_times'])
        else:
            stats['avg_processing_time'] = 0.0
            stats['std_processing_time'] = 0.0
            stats['fps_estimate'] = 0.0
        
        return stats
    
    def reset_stats(self):
        """통계 초기화"""
        self.stats = {
            'total_frames_processed': 0,
            'total_persons_detected': 0,
            'avg_persons_per_frame': 0.0,
            'processing_times': []
        }