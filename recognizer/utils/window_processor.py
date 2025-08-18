"""
윈도우 처리기

포즈 추정 → 트래킹 → 스코어링 결과를 윈도우 단위로 처리합니다.
"""

import numpy as np
import logging
from typing import List, Dict, Any
from .data_structure import FramePoses, WindowAnnotation


class SlidingWindowProcessor:
    """동적 버퍼 관리 슬라이딩 윈도우 처리기
    
    프레임 단위로 포즈 추정과 트래킹을 처리하고, 
    window_size만큼 프레임이 쌓이면 STGCN으로 분류 수행.
    stride 설정에 따라 효율적인 동적 버퍼 관리.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 처리 모드 설정
        self.processing_mode = config.get('processing_mode', 'realtime')  # 'realtime' or 'analysis'
        
        # 윈도우 설정
        self.window_size = config.get('window_size', 100)
        self.window_stride = config.get('window_stride', 50)
        
        # 자동 계산된 값들
        self.buffer_size = config.get('buffer_size', self.window_size + self.window_stride)  # 동적 버퍼 크기
        self.classification_delay = config.get('classification_delay', self.window_size)  # 첫 윈도우 완성 후 분류 시작
        
        # 동적 프레임 버퍼 (효율적인 순환 버퍼)
        self.frames_buffer: List[FramePoses] = []
        self.window_counter = 0
        self.processed_frame_count = 0
        
        # 윈도우 분류 결과 저장
        self.window_results: Dict[int, Dict] = {}  
        self.current_frame_idx = 0
        
        # 모드별 표시 설정
        if self.processing_mode == 'realtime':
            self.display_overlay_data = {
                'show_keypoints': config.get('show_keypoints', True),
                'show_tracking_ids': config.get('show_tracking_ids', True),
                'show_classification': False,  # classification_delay 후 활성화
                'show_composite_score': config.get('show_composite_score', False),  # 실시간에서는 비활성화
                'active_windows': []
            }
        else:  # analysis 모드
            self.display_overlay_data = {
                'show_keypoints': config.get('show_keypoints', True),
                'show_tracking_ids': config.get('show_tracking_ids', True),
                'show_classification': True,
                'show_composite_score': config.get('show_composite_score', True),  # 분석에서는 활성화
                'active_windows': []
            }
            
        logging.info(f"Initialized WindowProcessor: mode={self.processing_mode}, window_size={self.window_size}, stride={self.window_stride}, buffer_size={self.buffer_size}")
        
    def initialize(self) -> bool:
        """초기화"""
        self.frames_buffer = []
        self.window_counter = 0
        self.processed_frame_count = 0
        self.window_results = {}
        self.current_frame_idx = 0
        
        # 모드별 초기화
        if self.processing_mode == 'realtime':
            self.display_overlay_data['show_classification'] = False
            self.display_overlay_data['show_composite_score'] = False
        else:
            self.display_overlay_data['show_classification'] = True
            self.display_overlay_data['show_composite_score'] = True
            
        self.display_overlay_data['active_windows'] = []
        
        logging.info(f"WindowProcessor initialized: mode={self.processing_mode}")
        return True
    
    def reset(self):
        """윈도우 프로세서 상태 초기화 (새 비디오 처리 시 호출)"""
        logging.info("Resetting WindowProcessor state")
        
        # 버퍼 초기화
        self.frames_buffer = []
        
        # 카운터 초기화
        self.processed_frame_count = 0
        self.window_count = 0
        self.current_frame_idx = 0
        
        # 오버레이 데이터 초기화
        self.display_overlay_data = {
            'window_results': {},
            'active_windows': []
        }
        
        logging.info(f"WindowProcessor reset completed")
    
    def add_frame(self, frame: FramePoses) -> List[WindowAnnotation]:
        """동적 버퍼에 프레임 추가 및 윈도우 생성
        
        프레임 단위로 포즈 추정과 트래킹 처리 결과를 받아
        window_size만큼 쌓이면 STGCN으로 분류 수행
        
        Args:
            frame: 추가할 프레임 (포즈 추정 + 트래킹 완료)
            
        Returns:
            생성된 윈도우 리스트
        """
        if not frame:
            return []
        
        # 현재 프레임 상태 업데이트
        self.current_frame_idx = frame.frame_idx
        self.processed_frame_count += 1
        
        # 동적 버퍼에 프레임 추가
        self.frames_buffer.append(frame)
        
        logging.debug(f"Frame {frame.frame_idx} added, buffer: {len(self.frames_buffer)}, processed: {self.processed_frame_count}")
        
        # 실시간 모드에서 표시 상태 업데이트
        if self.processing_mode == 'realtime':
            self._update_realtime_overlay_status()
        
        # 윈도우 생성 확인 및 생성 (중복 방지 로직은 _create_windows_if_ready 내부에서 처리)
        windows = self._create_windows_if_ready()
        
        # 생성된 윈도우가 있으면 로그 출력
        if windows:
            for window in windows:
                logging.info(f"*** NEW WINDOW CREATED: {window.window_start}-{window.window_end} at frame {frame.frame_idx} ***")
        
        # 윈도우 생성 후 버퍼 크기 관리 (효율적인 메모리 사용)
        if len(self.frames_buffer) > self.buffer_size:
            # 오래된 프레임 제거 (stride만큼, 단 생성된 윈도우가 있을 때만)
            if windows:  # 윈도우가 생성되었을 때만 정리
                remove_count = min(self.window_stride, len(self.frames_buffer) - self.window_size)
                if remove_count > 0:
                    self.frames_buffer = self.frames_buffer[remove_count:]
                    logging.debug(f"Buffer cleanup after window creation: removed {remove_count} frames, new size: {len(self.frames_buffer)}")
        
        return windows
    
    def process_frames(self, frames: List[FramePoses]) -> List[WindowAnnotation]:
        """프레임들을 윈도우로 처리 (기존 호환성 유지)
        
        Args:
            frames: 처리할 프레임 리스트
            
        Returns:
            생성된 윈도우 리스트
        """
        if not frames:
            return []
        
        # 마지막 프레임만 처리 (중복 방지)
        return self.add_frame(frames[-1])
    
    def _create_windows_if_ready(self) -> List[WindowAnnotation]:
        """윈도우 생성 - 모드별 분기 처리"""
        if self.processing_mode == 'analysis':
            # 분석 모드는 한 번에 처리하므로 여기서는 빈 리스트 반환
            return []
        else:
            # 실시간 모드는 기존 로직 유지 (단순화)
            return self._create_realtime_windows()
    
    def _create_realtime_windows(self) -> List[WindowAnnotation]:
        """실시간 모드용 윈도우 생성 (기존 로직 단순화)"""
        windows = []
        
        if len(self.frames_buffer) < self.window_size:
            return windows
        
        # 실시간에서는 현재 버퍼로 하나의 윈도우만 생성 시도
        if len(self.frames_buffer) >= self.window_size:
            # 가장 최근 window_size개 프레임으로 윈도우 생성
            recent_frames = self.frames_buffer[-self.window_size:]
            
            # 연속성 확인
            sorted_frames = sorted(recent_frames, key=lambda x: x.frame_idx)
            is_consecutive = all(
                sorted_frames[i].frame_idx == sorted_frames[i-1].frame_idx + 1 
                for i in range(1, len(sorted_frames))
            )
            
            if is_consecutive:
                window = self._create_window_from_frames(sorted_frames, self.window_counter)
                if window:
                    window.window_start = sorted_frames[0].frame_idx
                    window.window_end = sorted_frames[-1].frame_idx
                    windows.append(window)
                    self.window_counter += 1
                    logging.info(f"Created realtime window: {window.window_start}-{window.window_end}")
        
        return windows
    
    def finalize_realtime_processing(self) -> List[WindowAnnotation]:
        """실시간 모드에서 비디오 끝날 때 남은 프레임들로 윈도우 생성"""
        windows = []
        
        if self.processing_mode != 'realtime':
            return windows
            
        if len(self.frames_buffer) < self.window_size:
            # 100프레임에 못 미치지만 최소한의 프레임이 있으면 패딩해서 윈도우 생성
            if len(self.frames_buffer) >= self.window_size * 0.4:  # 40% 이상이면 패딩
                logging.info(f"Creating padded window: {len(self.frames_buffer)} frames (padding to {self.window_size})")
                
                # 마지막 프레임을 복제해서 패딩
                padded_frames = self.frames_buffer.copy()
                last_frame = self.frames_buffer[-1] if self.frames_buffer else None
                
                if last_frame:
                    while len(padded_frames) < self.window_size:
                        # 마지막 프레임을 복사하되 frame_idx는 증가시킴
                        import copy
                        padded_frame = copy.deepcopy(last_frame)
                        padded_frame.frame_idx = padded_frames[-1].frame_idx + 1
                        padded_frames.append(padded_frame)
                    
                    # 연속성 확인 후 윈도우 생성
                    sorted_frames = sorted(padded_frames, key=lambda x: x.frame_idx)
                    window = self._create_window_from_frames(sorted_frames, self.window_counter)
                    if window:
                        window.window_start = sorted_frames[0].frame_idx
                        window.window_end = sorted_frames[-1].frame_idx
                        windows.append(window)
                        self.window_counter += 1
                        logging.info(f"Created final padded window: {window.window_start}-{window.window_end}")
        
        return windows
    
    def process_all_frames_analysis_mode(self, all_frames: List[FramePoses]) -> List[WindowAnnotation]:
        """분석 모드용 한 번에 모든 윈도우 처리
        
        Args:
            all_frames: 전체 프레임 리스트
            
        Returns:
            생성된 윈도우 리스트
        """
        if self.processing_mode != 'analysis':
            logging.warning("This method is only for analysis mode")
            return []
        
        if not all_frames:
            logging.warning("No frames provided")
            return []
        
        total_frames = len(all_frames)
        windows = []
        
        logging.info(f"Analysis mode: Processing {total_frames} frames with window_size={self.window_size}, stride={self.window_stride}")
        
        # 1. 윈도우 개수 및 범위 미리 계산
        window_ranges = []
        for start_idx in range(0, total_frames, self.window_stride):
            end_idx = start_idx + self.window_size
            
            if end_idx <= total_frames:
                # 완전한 윈도우
                window_ranges.append((start_idx, end_idx, 'complete'))
                logging.debug(f"Complete window planned: {start_idx}-{end_idx-1}")
            else:
                # 마지막 불완전 윈도우 - 40% 임계값 적용
                remaining_frames = total_frames - start_idx
                frame_ratio = remaining_frames / self.window_size
                
                if frame_ratio >= 0.4:  # 40% 이상이면 패딩하여 윈도우 생성
                    window_ranges.append((start_idx, total_frames, 'padding'))
                    logging.info(f"Padding window planned: {start_idx}-{total_frames-1} ({remaining_frames}/{self.window_size} frames, {frame_ratio:.1%} >= 40%)")
                else:
                    logging.info(f"Skipping final window: {start_idx}-{total_frames-1} ({remaining_frames}/{self.window_size} frames, {frame_ratio:.1%} < 40%)")
                break
        
        logging.info(f"Total windows to create: {len(window_ranges)}")
        
        # 2. 계획된 윈도우들 생성
        for window_idx, (start_idx, end_idx, window_type) in enumerate(window_ranges):
            try:
                if window_type == 'complete':
                    # 완전한 윈도우
                    window_frames = all_frames[start_idx:end_idx]
                    actual_end = start_idx + self.window_size - 1
                    
                elif window_type == 'padding':
                    # 패딩이 필요한 윈도우
                    window_frames = all_frames[start_idx:]
                    actual_frames_count = len(window_frames)
                    padding_needed = self.window_size - actual_frames_count
                    
                    # 0값 패딩 프레임 생성
                    if padding_needed > 0 and window_frames:
                        last_frame = window_frames[-1]
                        for i in range(padding_needed):
                            padding_frame = FramePoses(
                                frame_idx=last_frame.frame_idx + i + 1,
                                persons=[],  # 빈 persons 리스트 - 0값 패딩
                                timestamp=last_frame.timestamp + (i + 1) / 30.0,
                                image_shape=getattr(last_frame, 'image_shape', (480, 640)),
                                metadata=getattr(last_frame, 'metadata', {}).copy() if hasattr(last_frame, 'metadata') else {}
                            )
                            window_frames.append(padding_frame)
                        
                        logging.info(f"Added {padding_needed} padding frames to final window")
                    
                    actual_end = start_idx + self.window_size - 1
                
                # 윈도우 생성
                window = self._create_window_from_frames(window_frames, window_idx)
                
                if window:
                    window.window_start = start_idx
                    window.window_end = actual_end
                    windows.append(window)
                    
                    logging.info(f"Created window {window_idx}: frames {start_idx}-{actual_end} ({window_type})")
                else:
                    logging.error(f"Failed to create window {window_idx}: frames {start_idx}-{actual_end}")
                    
            except Exception as e:
                logging.error(f"Error creating window {window_idx}: {str(e)}")
                continue
        
        logging.info(f"Analysis mode completed: {len(windows)} windows created")
        return windows
    
    def _create_window_from_frames(self, frames: List[FramePoses], window_id: int) -> WindowAnnotation:
        """프레임들로부터 윈도우 생성
        
        Args:
            frames: 윈도우에 포함될 프레임들
            
        Returns:
            생성된 윈도우
        """
        if not frames:
            return None
        
        try:
            start_frame = frames[0].frame_idx
            end_frame = frames[-1].frame_idx
            
            # 디버깅: 프레임 인덱스 정보 출력
            frame_indices = [f.frame_idx for f in frames[:5]]  # 처음 5개만
            if len(frames) > 5:
                frame_indices.append("...")
                frame_indices.extend([f.frame_idx for f in frames[-2:]])  # 마지막 2개
            logging.info(f"Window frames: {frame_indices}, start={start_frame}, end={end_frame}")
        except (IndexError, AttributeError) as e:
            logging.warning(f"Error accessing frame indices: {str(e)}")
            return None
        
        # 키포인트 데이터 추출 (T, M, V, C) 형태로
        # T: 시간(프레임), M: 사람수, V: 키포인트수, C: 좌표차원
        try:
            person_counts = []
            for frame in frames:
                if hasattr(frame, 'persons') and frame.persons:
                    person_counts.append(len(frame.persons))
                else:
                    person_counts.append(0)
            
            max_persons = max(person_counts) if person_counts else 1
            if max_persons == 0:
                max_persons = 1
            logging.info(f"Calculated max_persons: {max_persons}, person_counts: {person_counts[:5]}...")
        except (ValueError, AttributeError) as e:
            logging.warning(f"Error calculating max persons: {str(e)}")
            max_persons = 1
            
        # 키포인트 데이터 초기화 (MMAction2 표준: [M, T, V, C])
        T = len(frames)  # 시간(프레임)
        V = 17  # 키포인트 수
        C = 2   # 좌표 차원 (x, y) - confidence는 별도
        
        keypoint_data = np.zeros((max_persons, T, V, C), dtype=np.float32)  # [M, T, V, C]
        keypoint_score_data = np.zeros((max_persons, T, V), dtype=np.float32)  # [M, T, V]
        
        # 프레임별 데이터 채우기
        for t, frame in enumerate(frames):
            try:
                if not hasattr(frame, 'persons') or not frame.persons:
                    continue
                    
                for m, person in enumerate(frame.persons):
                    if m >= max_persons:
                        break
                    
                    try:
                        if person is None or not hasattr(person, 'keypoints'):
                            continue
                            
                        if person.keypoints is not None:
                            if t < 5:  # 처음 5프레임만 출력
                                logging.info(f"Frame {t}, Person {m} keypoints: shape={getattr(person.keypoints, 'shape', len(person.keypoints) if hasattr(person.keypoints, '__len__') else 'unknown')}")
                            
                            # 키포인트가 (17, 3) 형태인 경우
                            if hasattr(person.keypoints, 'shape') and person.keypoints.shape == (17, 3):
                                # MMAction2 표준: [M, T, V, C] 형태로 저장
                                keypoint_data[m, t, :, :] = person.keypoints[:, :2]  # x, y 좌표만
                                keypoint_score_data[m, t, :] = person.keypoints[:, 2]  # confidence 별도
                                if t < 5:  # 처음 5프레임만 출력
                                    logging.info(f"Filled keypoints for Person {m}, Frame {t}, non-zero count: {np.count_nonzero(person.keypoints)}")
                            # 키포인트가 (51,) 형태인 경우 (17*3)
                            elif hasattr(person.keypoints, '__len__') and len(person.keypoints) == 51:
                                try:
                                    reshaped = np.array(person.keypoints).reshape(17, 3)
                                    keypoint_data[m, t, :, :] = reshaped[:, :2]  # x, y 좌표만
                                    keypoint_score_data[m, t, :] = reshaped[:, 2]  # confidence
                                    logging.debug(f"Reshaped and filled keypoints for person {m}, non-zero count: {np.count_nonzero(reshaped)}")
                                except (ValueError, AttributeError) as e:
                                    logging.warning(f"Error reshaping keypoints for person {m}: {str(e)}")
                            # 키포인트가 리스트 형태이고 17개 이상인 경우
                            elif hasattr(person.keypoints, '__len__') and len(person.keypoints) >= 17:
                                try:
                                    kpt_array = np.array(person.keypoints)
                                    if kpt_array.shape == (17, 3):
                                        keypoint_data[m, t, :, :] = kpt_array[:, :2]  # x, y 좌표만
                                        keypoint_score_data[m, t, :] = kpt_array[:, 2]  # confidence
                                        logging.debug(f"Converted list to array for person {m}, non-zero count: {np.count_nonzero(kpt_array)}")
                                except (ValueError, IndexError) as e:
                                    logging.warning(f"Error converting keypoints list for person {m}: {str(e)}")
                            else:
                                logging.warning(f"Unexpected keypoints format for person {m}: {type(person.keypoints)}, shape/len: {getattr(person.keypoints, 'shape', len(person.keypoints) if hasattr(person.keypoints, '__len__') else 'unknown')}")
                    except (IndexError, ValueError, AttributeError) as e:
                        logging.warning(f"Error processing person {m} in frame {t}: {str(e)}")
                        continue
            except Exception as e:
                logging.warning(f"Error processing frame {t}: {str(e)}")
                continue
        
        # 이미지 크기 추정 (첫 번째 프레임에서)
        try:
            img_shape = frames[0].image_shape if hasattr(frames[0], 'image_shape') and frames[0].image_shape else (480, 640)
        except (IndexError, AttributeError):
            img_shape = (480, 640)
        
        # 키포인트 데이터 통계 출력 (MMAction2 표준 형태)
        logging.info(f"Window keypoint_data shape: {keypoint_data.shape} (MMAction2 standard: [M, T, V, C])")
        logging.info(f"Window keypoint_data non-zero ratio: {np.count_nonzero(keypoint_data) / keypoint_data.size:.4f}")
        logging.info(f"Window keypoint_data value range: {np.min(keypoint_data[keypoint_data!=0]) if np.any(keypoint_data!=0) else 0:.3f} ~ {np.max(keypoint_data):.3f}")
        logging.info(f"Window keypoint_score shape: {keypoint_score_data.shape} (MMAction2 standard: [M, T, V])")
        
        # WindowAnnotation 생성
        window = WindowAnnotation(
            window_idx=window_id,
            start_frame=start_frame,
            end_frame=end_frame,
            keypoint=keypoint_data,
            keypoint_score=keypoint_score_data,
            frame_dir="",  # 실시간 처리에서는 불필요
            img_shape=img_shape,
            original_shape=img_shape,
            total_frames=len(frames),
            label=0  # 분류 전 기본값
        )
        
        # 원본 프레임 데이터도 저장 (필요시 참조용)
        window.frame_data = frames
        
        return window
    
    def _update_realtime_overlay_status(self):
        """실시간 모드 오버레이 표시 상태 업데이트"""
        current_frame = self.current_frame_idx
        
        # classification_delay 후에 분류 결과 표시 시작
        if self.processed_frame_count >= self.classification_delay:
            self.display_overlay_data['show_classification'] = True
        
        # Previous/Current 윈도우 표시 로직
        active_windows = []
        
        if len(self.window_results) > 0:
            available_windows = sorted(self.window_results.keys())
            
            # 4번째 윈도우 이후부터는 항상 Previous/Current 표시
            if len(available_windows) >= 4:
                active_windows = available_windows[-2:]  # 마지막 2개 (Previous, Current)
            elif len(available_windows) >= 2:
                active_windows = available_windows[-2:]  # 기존 2개
            elif len(available_windows) == 1:
                active_windows = available_windows[-1:]  # Current만
        
        self.display_overlay_data['active_windows'] = active_windows
        
        logging.debug(f"Frame {current_frame}: processed={self.processed_frame_count}, show_classification={self.display_overlay_data['show_classification']}, active_windows={active_windows}")
    
    def add_window_result(self, window_id: int, classification_result: Dict) -> None:
        """윈도우 분류 결과 추가
        
        Args:
            window_id: 윈도우 번호
            classification_result: 분류 결과 (label, confidence 등)
        """
        # classification_result에서 실제 프레임 범위 가져오기
        start_frame = classification_result.get('window_start', window_id * self.window_stride)
        end_frame = classification_result.get('window_end', window_id * self.window_stride + self.window_size - 1)
        
        self.window_results[window_id] = {
            'window_id': window_id,
            'classification': classification_result,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'timestamp': classification_result.get('timestamp', None)
        }
        
        logging.info(f"Added window {window_id} result: {classification_result}")
        
        # 오버레이 상태 업데이트
        if self.processing_mode == 'realtime':
            self._update_realtime_overlay_status()
        # 분석 모드에서는 오버레이 상태 업데이트 불필요
    
    def get_current_overlay_data(self) -> Dict:
        """현재 프레임에 대한 오버레이 데이터 반환
        
        Returns:
            현재 표시할 오버레이 정보
        """
        overlay_data = {
            'frame_idx': self.current_frame_idx,
            'show_keypoints': self.display_overlay_data['show_keypoints'],
            'show_tracking_ids': self.display_overlay_data['show_tracking_ids'],
            'show_classification': self.display_overlay_data['show_classification'],
            'show_composite_score': self.display_overlay_data['show_composite_score'],
            'window_results': []
        }
        
        # 현재 활성 윈도우들의 결과 추가
        for window_id in self.display_overlay_data['active_windows']:
            if window_id in self.window_results:
                overlay_data['window_results'].append(self.window_results[window_id])
        
        return overlay_data
    
    def get_overlay_info_for_frame(self, frame_idx: int) -> Dict:
        """특정 프레임에 대한 오버레이 정보 반환 (분석 결과 기반)
        
        Args:
            frame_idx: 프레임 번호
            
        Returns:
            해당 프레임의 오버레이 정보
        """
        overlay_info = {
            'frame_idx': frame_idx,
            'keypoints_data': None,
            'tracking_ids': [],
            'classification_results': [],
            'composite_scores': []
        }
        
        # 해당 프레임이 포함된 윈도우들 찾기
        relevant_windows = []
        for window_id, result in self.window_results.items():
            if result['start_frame'] <= frame_idx <= result['end_frame']:
                relevant_windows.append(result)
        
        overlay_info['classification_results'] = relevant_windows
        
        # 프레임 버퍼에서 해당 프레임 찾기
        for frame in self.frames_buffer:
            if frame.frame_idx == frame_idx:
                # 관절 포인트 데이터 추출
                if hasattr(frame, 'persons') and frame.persons:
                    keypoints_data = []
                    tracking_ids = []
                    
                    for person in frame.persons:
                        if person and hasattr(person, 'keypoints'):
                            keypoints_data.append(person.keypoints)
                            
                        if person and hasattr(person, 'track_id'):
                            tracking_ids.append(person.track_id)
                    
                    overlay_info['keypoints_data'] = keypoints_data
                    overlay_info['tracking_ids'] = tracking_ids
                break
        
        return overlay_info
    
    def should_show_composite_score(self, frame_idx: int) -> bool:
        """복합점수 표시 여부 결정
        
        실시간에서는 복합점수를 표시하지 않음 (윈도우 완성 후에만 가능)
        
        Args:
            frame_idx: 프레임 번호
            
        Returns:
            복합점수 표시 여부
        """
        # 실시간 처리에서는 복합점수 표시 안함
        return False
    
    def get_frame_display_mode(self, frame_idx: int) -> str:
        """프레임 표시 모드 결정
        
        Args:
            frame_idx: 프레임 번호
            
        Returns:
            표시 모드 ('realtime', 'analysis', 'full')
        """
        if frame_idx == self.current_frame_idx:
            # 현재 처리 중인 프레임 (실시간)
            return 'realtime'
        elif frame_idx in [result['start_frame'] for result in self.window_results.values()]:
            # 분석 완료된 윈도우의 시작 프레임
            return 'analysis'
        else:
            # 일반 프레임
            return 'full'
    
    def finalize_processing(self) -> List[WindowAnnotation]:
        """마지막 윈도우 처리 - 부족한 프레임은 0값으로 패딩"""
        windows = []
        
        # 버퍼에 남은 프레임들로 마지막 윈도우 생성
        remaining_frames = len(self.frames_buffer)
        if remaining_frames > 0:
            # 마지막 윈도우가 생성되지 않은 영역이 있는지 확인
            last_window_start = self.window_counter * self.window_stride
            
            # 마지막 윈도우를 생성해야 하는 조건:
            # 1. 버퍼에 윈도우 시작점 이후의 프레임이 있는 경우
            # 2. 또는 첫 번째 윈도우도 생성되지 않은 경우 (짧은 비디오)
            should_create_final = (remaining_frames > last_window_start) or (self.window_counter == 0 and remaining_frames > 0)
            
            if should_create_final:
                logging.info(f"Creating final window with padding: {remaining_frames} frames available, need {self.window_size}, window_counter={self.window_counter}")
                
                # 마지막 윈도우 프레임들 추출
                if self.window_counter == 0:
                    # 첫 번째 윈도우인 경우 처음부터
                    final_frames = self.frames_buffer[:]
                else:
                    # 일반적인 경우 스트라이드 위치부터
                    final_frames = self.frames_buffer[last_window_start:]
                
                # 부족한 프레임 수 계산 및 40% 임계값 조건 확인
                actual_frames_count = len(final_frames)
                padding_needed = self.window_size - actual_frames_count
                
                # 실제 프레임이 윈도우 크기의 40% 미만이면 윈도우 생성하지 않음
                frame_ratio = actual_frames_count / self.window_size
                min_frame_ratio = 0.4  # 40% 임계값
                
                if frame_ratio < min_frame_ratio:
                    logging.info(f"Skipping final window: {actual_frames_count}/{self.window_size} frames ({frame_ratio:.1%}) < {min_frame_ratio:.0%} threshold")
                    return windows  # 마지막 윈도우 생성하지 않고 종료
                
                if padding_needed > 0:
                    # 40% 이상이므로 패딩하여 윈도우 생성
                    logging.info(f"Creating final window with padding: {actual_frames_count}/{self.window_size} frames ({frame_ratio:.1%}) >= {min_frame_ratio:.0%} threshold")
                    
                    # 마지막 프레임을 복제하여 패딩 생성
                    last_frame = final_frames[-1] if final_frames else self.frames_buffer[-1]
                    
                    # 패딩 프레임들 생성 (빈 persons 리스트로 0값 패딩)
                    for i in range(padding_needed):
                        padding_frame = FramePoses(
                            frame_idx=last_frame.frame_idx + i + 1,
                            persons=[],  # 빈 persons 리스트 - 0값 패딩 효과
                            timestamp=last_frame.timestamp + (i + 1) / 30.0,
                            image_shape=getattr(last_frame, 'image_shape', (480, 640)),
                            metadata=getattr(last_frame, 'metadata', {}).copy() if hasattr(last_frame, 'metadata') else {}
                        )
                        final_frames.append(padding_frame)
                    
                    logging.info(f"Added {padding_needed} zero-padding frames for final window (total frames: {len(final_frames)})")
                else:
                    logging.info(f"Final window has sufficient frames: {actual_frames_count}/{self.window_size} frames, no padding needed")
                
                # 최종 윈도우 생성
                try:
                    window = self._create_window_from_frames(final_frames, self.window_counter)
                    if window:
                        windows.append(window)
                        self.window_counter += 1
                        logging.info(f"Created final padded window {self.window_counter} with frames {final_frames[0].frame_idx}-{final_frames[-1].frame_idx}")
                except Exception as e:
                    logging.warning(f"Error creating final window: {str(e)}")
        
        return windows
    
    def get_processing_mode(self) -> str:
        """현재 처리 모드 반환"""
        return self.processing_mode
    
    def set_processing_mode(self, mode: str) -> None:
        """처리 모드 변경
        
        Args:
            mode: 'realtime' 또는 'analysis'
        """
        if mode in ['realtime', 'analysis']:
            self.processing_mode = mode
            
            # 모드에 따른 표시 설정 업데이트
            if mode == 'realtime':
                self.display_overlay_data['show_composite_score'] = False
                if self.processed_frame_count < self.classification_delay:
                    self.display_overlay_data['show_classification'] = False
            else:  # analysis
                self.display_overlay_data['show_composite_score'] = True
                self.display_overlay_data['show_classification'] = True
                
            logging.info(f"Processing mode changed to: {mode}")
        else:
            logging.warning(f"Invalid processing mode: {mode}. Use 'realtime' or 'analysis'")
    
    def get_buffer_status(self) -> Dict:
        """버퍼 상태 정보 반환"""
        return {
            'buffer_size': len(self.frames_buffer),
            'max_buffer_size': self.buffer_size,
            'window_counter': self.window_counter,
            'processed_frame_count': self.processed_frame_count,
            'current_frame_idx': self.current_frame_idx,
            'windows_created': len(self.window_results),
            'processing_mode': self.processing_mode
        }
    
    def cleanup(self):
        """정리"""
        self.frames_buffer = []
        self.window_counter = 0
        self.processed_frame_count = 0
        self.window_results = {}
        self.current_frame_idx = 0