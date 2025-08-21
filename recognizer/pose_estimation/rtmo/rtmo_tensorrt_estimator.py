#!/usr/bin/env python3
"""
RTMO TensorRT 포즈 추정기 구현

TensorRT를 사용한 고성능 RTMO 포즈 추정기입니다.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple
import logging

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    logging.warning("TensorRT not available")

try:
    from pose_estimation.base import BasePoseEstimator
    from utils.data_structure import PersonPose, FramePoses, PoseEstimationConfig
except ImportError:
    from ..base import BasePoseEstimator
    from ...utils.data_structure import PersonPose, FramePoses, PoseEstimationConfig


class RTMOTensorRTEstimator(BasePoseEstimator):
    """RTMO TensorRT 포즈 추정기 구현"""
    
    def __init__(self, config: PoseEstimationConfig):
        """
        Args:
            config: RTMO TensorRT 포즈 추정 설정
        """
        super().__init__(config)
        
        if not TENSORRT_AVAILABLE:
            raise RuntimeError("TensorRT is required for TensorRT pose estimation")
        
        # TensorRT 특화 설정
        self.engine_path = config.model_path  # TensorRT engine 경로
        self.model_input_size = getattr(config, 'model_input_size', (640, 640))
        self.mean = getattr(config, 'mean', None)
        self.std = getattr(config, 'std', None)
        self.to_openpose = getattr(config, 'to_openpose', False)
        self.fp16_mode = getattr(config, 'fp16_mode', False)
        
        # TensorRT 엔진 및 컨텍스트
        self.engine = None
        self.context = None
        self.bindings = []
        self.host_inputs = []
        self.host_outputs = []
        self.cuda_inputs = []
        self.cuda_outputs = []
        self.stream = None
        
        # 입출력 정보
        self.input_shape = None
        self.output_shapes = []
        
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
        """TensorRT 엔진 초기화"""
        try:
            if self.is_initialized:
                return True
                
            if not os.path.exists(self.engine_path):
                raise FileNotFoundError(f"TensorRT engine not found: {self.engine_path}")
            
            # TensorRT 로거 설정
            trt_logger = trt.Logger(trt.Logger.WARNING)
            
            # 엔진 로드
            with open(self.engine_path, 'rb') as f:
                engine_data = f.read()
            
            runtime = trt.Runtime(trt_logger)
            self.engine = runtime.deserialize_cuda_engine(engine_data)
            
            if self.engine is None:
                raise RuntimeError("Failed to load TensorRT engine")
            
            # 실행 컨텍스트 생성
            self.context = self.engine.create_execution_context()
            
            # CUDA 스트림 생성
            self.stream = cuda.Stream()
            
            # 바인딩 정보 설정
            self._setup_bindings()
            
            self.is_initialized = True
            print(f"TensorRT engine initialized successfully: {self.engine_path}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize TensorRT engine: {str(e)}")
            self.is_initialized = False
            return False
    
    def _setup_bindings(self):
        """입출력 바인딩 설정"""
        self.bindings = []
        self.host_inputs = []
        self.host_outputs = []
        self.cuda_inputs = []
        self.cuda_outputs = []
        self.output_shapes = []
        
        for i in range(self.engine.num_bindings):
            binding_name = self.engine.get_binding_name(i)
            binding_shape = self.engine.get_binding_shape(i)
            binding_dtype = trt.nptype(self.engine.get_binding_dtype(i))
            binding_size = trt.volume(binding_shape)
            
            if self.engine.binding_is_input(i):
                # 입력 바인딩
                self.input_shape = binding_shape
                host_mem = cuda.pagelocked_empty(binding_size, binding_dtype)
                cuda_mem = cuda.mem_alloc(host_mem.nbytes)
                
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
                
                print(f"Input {i}: {binding_name} - Shape: {binding_shape}, Type: {binding_dtype}")
            else:
                # 출력 바인딩
                self.output_shapes.append(binding_shape)
                host_mem = cuda.pagelocked_empty(binding_size, binding_dtype)
                cuda_mem = cuda.mem_alloc(host_mem.nbytes)
                
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)
                
                print(f"Output {i}: {binding_name} - Shape: {binding_shape}, Type: {binding_dtype}")
            
            self.bindings.append(int(cuda_mem))
    
    def extract_poses(self, frame: np.ndarray, frame_idx: int = 0) -> List[PersonPose]:
        """단일 프레임에서 포즈 추출"""
        if not self.ensure_initialized():
            raise RuntimeError("Failed to initialize TensorRT engine")
        
        if not self.validate_frame(frame):
            return []
        
        try:
            import time
            start_time = time.time()
            
            # TensorRT 추론
            keypoints, scores = self._inference_tensorrt(frame)
            
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
            logging.error(f"Error in TensorRT pose extraction for frame {frame_idx}: {str(e)}")
            return []
    
    def _inference_tensorrt(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """TensorRT 엔진 추론 수행"""
        # 전처리
        preprocessed_img, ratio = self._preprocess(image)
        
        # 추론
        outputs = self._inference(preprocessed_img)
        
        # 후처리
        keypoints, scores = self._postprocess(outputs, ratio)
        
        # OpenPose 형식으로 변환 (옵션)
        if self.to_openpose:
            keypoints, scores = self._convert_coco_to_openpose(keypoints, scores)
        
        return keypoints, scores
    
    def _preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        """전처리: 이미지 리사이즈 및 패딩"""
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
        """TensorRT 추론 실행"""
        # 입력 형식 변환: (H, W, C) -> (C, H, W)
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        
        # 입력 데이터를 GPU 메모리로 복사
        np.copyto(self.host_inputs[0], img.ravel())
        cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
        
        # 추론 실행
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # 출력 데이터를 CPU 메모리로 복사
        outputs = []
        for i in range(len(self.host_outputs)):
            cuda.memcpy_dtoh_async(self.host_outputs[i], self.cuda_outputs[i], self.stream)
            self.stream.synchronize()
            
            # 출력 형태로 변환
            output_shape = self.output_shapes[i]
            output = self.host_outputs[i].reshape(output_shape)
            outputs.append(output)
        
        return outputs
    
    def _postprocess(self, outputs: List[np.ndarray], ratio: float) -> Tuple[np.ndarray, np.ndarray]:
        """후처리: NMS 적용 및 좌표 복원"""
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
                logging.warning(f"Error converting TensorRT result {person_idx}: {str(e)}")
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
        
        print(f"Extracting poses from: {video_path} using TensorRT")
        
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
        
        with tqdm(total=total_frames, desc="Processing frames (TensorRT)") as pbar:
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
        
        print(f"Extracted poses from {len(frame_poses_list)} frames using TensorRT")
        print(f"Average persons per frame: {self.stats['avg_persons_per_frame']:.2f}")
        
        return frame_poses_list
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        info = {
            'model_name': 'RTMO-TensorRT',
            'engine_path': self.engine_path,
            'model_input_size': self.model_input_size,
            'fp16_mode': self.fp16_mode,
            'device': self.device,
            'score_threshold': self.score_threshold,
            'nms_threshold': self.nms_threshold,
            'keypoint_threshold': self.keypoint_threshold,
            'max_detections': self.max_detections,
            'to_openpose': self.to_openpose,
            'is_initialized': self.is_initialized,
            'statistics': self.stats.copy()
        }
        
        if self.engine:
            try:
                info['input_shape'] = self.input_shape
                info['output_shapes'] = self.output_shapes
                info['num_bindings'] = self.engine.num_bindings
                info['max_batch_size'] = self.engine.max_batch_size
            except:
                pass
        
        return info
    
    def cleanup(self):
        """리소스 정리"""
        super().cleanup()
        
        # CUDA 메모리 해제
        for cuda_mem in self.cuda_inputs + self.cuda_outputs:
            cuda_mem.free()
        
        self.cuda_inputs.clear()
        self.cuda_outputs.clear()
        self.host_inputs.clear()
        self.host_outputs.clear()
        
        if self.stream:
            self.stream.free()
            self.stream = None
        
        if self.context:
            del self.context
            self.context = None
        
        if self.engine:
            del self.engine
            self.engine = None
        
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