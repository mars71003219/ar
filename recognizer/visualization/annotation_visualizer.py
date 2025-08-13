"""
어노테이션 시각화 도구

어노테이션 작업 및 데이터 검증을 위한 시각화 도구입니다.
"""

import cv2
import numpy as np
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path

from ..utils.data_structure import PersonPose, FramePoses
from ..pipelines.annotation_pipeline import AnnotationData
from .pose_visualizer import PoseVisualizer


class AnnotationVisualizer:
    """어노테이션 시각화 클래스"""
    
    def __init__(self):
        """초기화"""
        self.pose_visualizer = PoseVisualizer(
            show_bbox=True,
            show_keypoints=True,
            show_skeleton=True,
            show_track_id=True,
            show_confidence=True
        )
    
    def visualize_annotation_data(self, annotation_data: AnnotationData, 
                                video_path: str,
                                output_path: Optional[str] = None,
                                fps: float = 30.0) -> bool:
        """어노테이션 데이터 시각화
        
        Args:
            annotation_data: 어노테이션 데이터
            video_path: 원본 비디오 경로
            output_path: 출력 경로 (None이면 화면 표시)
            fps: 출력 FPS
            
        Returns:
            성공 여부
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        
        # 비디오 정보
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 출력 설정
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        try:
            frame_idx = 0
            annotation_idx = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 해당 프레임의 어노테이션 데이터 찾기
                current_annotation = None
                if (annotation_idx < len(annotation_data.frame_annotations) and 
                    annotation_data.frame_annotations[annotation_idx]['frame_idx'] == frame_idx):
                    current_annotation = annotation_data.frame_annotations[annotation_idx]
                    annotation_idx += 1
                
                # 시각화
                if current_annotation:
                    vis_frame = self._draw_annotation_on_frame(frame, current_annotation, frame_idx)
                else:
                    vis_frame = frame.copy()
                    self._draw_no_annotation_info(vis_frame, frame_idx)
                
                # 비디오 정보 및 라벨 표시
                self._draw_video_info(vis_frame, annotation_data, frame_idx)
                
                # 출력
                if writer:
                    writer.write(vis_frame)
                else:
                    cv2.imshow('Annotation Visualization', vis_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord(' '):  # 스페이스바로 일시정지
                        cv2.waitKey(0)
                
                frame_idx += 1
            
            return True
            
        except Exception as e:
            print(f"Error in annotation visualization: {str(e)}")
            return False
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if not output_path:
                cv2.destroyAllWindows()
    
    def _draw_annotation_on_frame(self, frame: np.ndarray, annotation: Dict[str, Any], frame_idx: int) -> np.ndarray:
        """프레임에 어노테이션 그리기"""
        vis_frame = frame.copy()
        
        # FramePoses 객체 생성
        persons = []
        for person_data in annotation['persons']:
            person = PersonPose(
                bbox=person_data.get('bbox', []),
                keypoints=np.array(person_data.get('keypoints', [])),
                score=person_data.get('score', 0.0),
                track_id=person_data.get('track_id')
            )
            persons.append(person)
        
        frame_poses = FramePoses(
            frame_idx=frame_idx,
            timestamp=annotation.get('timestamp', 0.0),
            persons=persons
        )
        
        # 포즈 시각화
        vis_frame = self.pose_visualizer.visualize_frame(vis_frame, frame_poses)
        
        return vis_frame
    
    def _draw_no_annotation_info(self, frame: np.ndarray, frame_idx: int):
        """어노테이션 없는 프레임 정보 표시"""
        text = f"Frame {frame_idx}: No annotation data"
        
        # 텍스트 배경
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(frame, (10, 10), (text_size[0] + 20, text_size[1] + 20), 
                     (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (text_size[0] + 20, text_size[1] + 20), 
                     (128, 128, 128), 1)
        
        # 텍스트
        cv2.putText(frame, text, (15, text_size[1] + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
    
    def _draw_video_info(self, frame: np.ndarray, annotation_data: AnnotationData, current_frame: int):
        """비디오 정보 및 라벨 표시"""
        height = frame.shape[0]
        
        # 비디오 정보
        video_info = annotation_data.video_info
        info_lines = [
            f"Video: {Path(video_info['filename']).stem}",
            f"Label: {video_info.get('label', 'Unknown')}",
            f"Frame: {current_frame}/{video_info['total_frames']}",
            f"Duration: {video_info.get('duration', 0):.1f}s"
        ]
        
        # 처리 통계
        stats = annotation_data.processing_stats
        stats_lines = [
            f"Tracks: {stats.get('unique_tracks', 0)}",
            f"Valid Frames: {stats.get('valid_frames', 0)}",
            f"Avg Persons: {stats.get('avg_persons_per_frame', 0):.1f}"
        ]
        
        # 정보 패널 그리기
        panel_width = 250
        panel_height = len(info_lines + stats_lines) * 25 + 40
        
        # 패널 배경
        cv2.rectangle(frame, (frame.shape[1] - panel_width - 10, height - panel_height - 10),
                     (frame.shape[1] - 10, height - 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (frame.shape[1] - panel_width - 10, height - panel_height - 10),
                     (frame.shape[1] - 10, height - 10), (255, 255, 255), 1)
        
        # 텍스트 그리기
        y_offset = height - panel_height + 15
        
        for line in info_lines:
            cv2.putText(frame, line, (frame.shape[1] - panel_width + 5, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
        
        # 구분선
        cv2.line(frame, (frame.shape[1] - panel_width + 5, y_offset - 10),
                (frame.shape[1] - 15, y_offset - 10), (128, 128, 128), 1)
        
        for line in stats_lines:
            cv2.putText(frame, line, (frame.shape[1] - panel_width + 5, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 25
    
    def create_annotation_summary(self, annotation_data: AnnotationData, 
                                output_size: Tuple[int, int] = (1200, 800)) -> np.ndarray:
        """어노테이션 요약 이미지 생성
        
        Args:
            annotation_data: 어노테이션 데이터
            output_size: 출력 크기
            
        Returns:
            요약 이미지
        """
        width, height = output_size
        summary_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 제목
        title = f"Annotation Summary - {Path(annotation_data.video_info['filename']).stem}"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
        cv2.putText(summary_image, title, 
                   ((width - title_size[0]) // 2, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # 비디오 정보
        video_info = annotation_data.video_info
        info_text = [
            f"Label: {video_info.get('label', 'Unknown')}",
            f"Resolution: {video_info['width']}x{video_info['height']}",
            f"FPS: {video_info['fps']:.1f}",
            f"Total Frames: {video_info['total_frames']}",
            f"Duration: {video_info.get('duration', 0):.1f}s"
        ]
        
        y_offset = 100
        for text in info_text:
            cv2.putText(summary_image, text, (50, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
            y_offset += 30
        
        # 처리 통계
        stats = annotation_data.processing_stats
        stats_text = [
            f"Processing Time: {stats.get('processing_time', 0):.2f}s",
            f"Total Frames: {stats.get('total_frames', 0)}",
            f"Filtered Frames: {stats.get('filtered_frames', 0)}",
            f"Valid Frames: {stats.get('valid_frames', 0)}",
            f"Total Person Detections: {stats.get('total_persons', 0)}",
            f"Unique Tracks: {stats.get('unique_tracks', 0)}",
            f"Avg Persons per Frame: {stats.get('avg_persons_per_frame', 0):.2f}"
        ]
        
        y_offset = 100
        for text in stats_text:
            cv2.putText(summary_image, text, (width // 2 + 50, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
            y_offset += 30
        
        # 트랙 요약
        track_summary = annotation_data.track_summary
        if track_summary:
            y_offset += 50
            cv2.putText(summary_image, "Track Summary:", (50, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
            y_offset += 40
            
            # 상위 5개 트랙 표시
            sorted_tracks = sorted(track_summary.items(), 
                                 key=lambda x: x[1]['frame_count'], reverse=True)
            
            for i, (track_id, track_info) in enumerate(sorted_tracks[:5]):
                track_text = (f"Track {track_id}: {track_info['frame_count']} frames, "
                            f"Duration: {track_info['duration']} frames, "
                            f"Avg Score: {track_info['avg_score']:.3f}")
                
                cv2.putText(summary_image, track_text, (70, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 255, 150), 1)
                y_offset += 25
        
        # 프레임별 person 수 그래프 (간단한 버전)
        if annotation_data.frame_annotations:
            graph_y = height - 150
            graph_height = 100
            graph_width = width - 100
            
            # 프레임별 person 수 계산
            persons_per_frame = []
            frame_indices = []
            
            for annotation in annotation_data.frame_annotations:
                persons_per_frame.append(len(annotation['persons']))
                frame_indices.append(annotation['frame_idx'])
            
            if persons_per_frame:
                max_persons = max(persons_per_frame)
                
                # 그래프 제목
                cv2.putText(summary_image, "Persons per Frame", (50, graph_y - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
                
                # 그래프 그리기
                for i in range(len(persons_per_frame) - 1):
                    if len(frame_indices) > 1:
                        x1 = int(50 + (frame_indices[i] * graph_width) / max(frame_indices))
                        x2 = int(50 + (frame_indices[i+1] * graph_width) / max(frame_indices))
                    else:
                        x1 = x2 = 50
                    
                    y1 = int(graph_y - (persons_per_frame[i] * graph_height) / max_persons) if max_persons > 0 else graph_y
                    y2 = int(graph_y - (persons_per_frame[i+1] * graph_height) / max_persons) if max_persons > 0 else graph_y
                    
                    cv2.line(summary_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 축 그리기
                cv2.line(summary_image, (50, graph_y), (50 + graph_width, graph_y), (255, 255, 255), 1)  # X축
                cv2.line(summary_image, (50, graph_y), (50, graph_y - graph_height), (255, 255, 255), 1)  # Y축
                
                # 축 라벨
                cv2.putText(summary_image, "0", (45, graph_y + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(summary_image, str(max_persons), (30, graph_y - graph_height + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return summary_image
    
    def compare_annotations(self, annotation_data1: AnnotationData, 
                          annotation_data2: AnnotationData,
                          output_size: Tuple[int, int] = (1400, 800)) -> np.ndarray:
        """두 어노테이션 비교 시각화
        
        Args:
            annotation_data1: 첫 번째 어노테이션
            annotation_data2: 두 번째 어노테이션
            output_size: 출력 크기
            
        Returns:
            비교 이미지
        """
        width, height = output_size
        compare_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 제목
        title = "Annotation Comparison"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
        cv2.putText(compare_image, title, 
                   ((width - title_size[0]) // 2, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # 두 어노테이션 비교 정보
        stats1 = annotation_data1.processing_stats
        stats2 = annotation_data2.processing_stats
        
        # 왼쪽: 첫 번째 어노테이션
        cv2.putText(compare_image, "Annotation 1", (50, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        info1 = [
            f"Video: {Path(annotation_data1.video_info['filename']).stem}",
            f"Label: {annotation_data1.video_info.get('label', 'Unknown')}",
            f"Valid Frames: {stats1.get('valid_frames', 0)}",
            f"Unique Tracks: {stats1.get('unique_tracks', 0)}",
            f"Total Persons: {stats1.get('total_persons', 0)}",
            f"Avg Persons/Frame: {stats1.get('avg_persons_per_frame', 0):.2f}"
        ]
        
        y_offset = 130
        for text in info1:
            cv2.putText(compare_image, text, (70, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            y_offset += 30
        
        # 오른쪽: 두 번째 어노테이션
        cv2.putText(compare_image, "Annotation 2", (width // 2 + 50, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        
        info2 = [
            f"Video: {Path(annotation_data2.video_info['filename']).stem}",
            f"Label: {annotation_data2.video_info.get('label', 'Unknown')}",
            f"Valid Frames: {stats2.get('valid_frames', 0)}",
            f"Unique Tracks: {stats2.get('unique_tracks', 0)}",
            f"Total Persons: {stats2.get('total_persons', 0)}",
            f"Avg Persons/Frame: {stats2.get('avg_persons_per_frame', 0):.2f}"
        ]
        
        y_offset = 130
        for text in info2:
            cv2.putText(compare_image, text, (width // 2 + 70, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            y_offset += 30
        
        # 구분선
        cv2.line(compare_image, (width // 2, 80), (width // 2, height - 50), (128, 128, 128), 2)
        
        return compare_image
    
    def visualize_stage2_pkl(self, pkl_path: Union[str, Path], 
                            video_path: str,
                            output_path: Optional[str] = None,
                            fps: float = 30.0,
                            show_windows: bool = True,
                            show_scores: bool = True) -> bool:
        """Stage 2 pkl 파일 시각화
        
        Args:
            pkl_path: Stage 2에서 생성된 pkl 파일 경로
            video_path: 원본 비디오 경로
            output_path: 출력 경로 (None이면 화면 표시)
            fps: 출력 FPS
            show_windows: 윈도우 정보 표시 여부
            show_scores: 복합점수 표시 여부
            
        Returns:
            성공 여부
        """
        try:
            # pkl 파일 로드
            with open(pkl_path, 'rb') as f:
                stage2_data = pickle.load(f)
            
            logging.info(f"Loaded Stage 2 data: {stage2_data.get('video_name', 'Unknown')}")
            logging.info(f"Total windows: {stage2_data.get('num_windows', 0)}")
            
            # 비디오 열기
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logging.error(f"Cannot open video: {video_path}")
                return False
            
            # 비디오 정보
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 출력 설정
            writer = None
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                logging.info(f"Output video will be saved to: {output_path}")
            
            # Stage 2 데이터에서 프레임별 어노테이션 추출
            frame_annotations = self._extract_frame_annotations_from_stage2(stage2_data)
            
            frame_idx = 0
            annotation_lookup = {ann['frame_idx']: ann for ann in frame_annotations}
            
            while cap.isOpened() and frame_idx < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 현재 프레임의 어노테이션 찾기
                current_annotation = annotation_lookup.get(frame_idx)
                
                # 시각화
                if current_annotation:
                    vis_frame = self._draw_stage2_annotation_on_frame(
                        frame, current_annotation, stage2_data, frame_idx, 
                        show_windows, show_scores
                    )
                else:
                    vis_frame = frame.copy()
                    self._draw_no_annotation_info(vis_frame, frame_idx)
                
                # Stage 2 정보 표시
                self._draw_stage2_info(vis_frame, stage2_data, frame_idx, total_frames)
                
                # 출력
                if writer:
                    writer.write(vis_frame)
                else:
                    cv2.imshow('Stage 2 Visualization', vis_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord(' '):  # 스페이스바로 일시정지
                        cv2.waitKey(0)
                    elif key == ord('s') and show_scores:  # 's'키로 점수 토글
                        show_scores = not show_scores
                    elif key == ord('w') and show_windows:  # 'w'키로 윈도우 정보 토글
                        show_windows = not show_windows
                
                frame_idx += 1
            
            logging.info(f"Visualization completed. Processed {frame_idx} frames.")
            return True
            
        except Exception as e:
            logging.error(f"Error in Stage 2 visualization: {str(e)}")
            return False
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if not output_path:
                cv2.destroyAllWindows()
    
    def _extract_frame_annotations_from_stage2(self, stage2_data: Dict) -> List[Dict]:
        """Stage 2 데이터에서 프레임별 어노테이션 추출"""
        frame_annotations = []
        windows = stage2_data.get('windows', [])
        
        for window in windows:
            annotation = window.get('annotation', {})
            if 'annotation' not in annotation:
                continue
                
            start_frame = window.get('start_frame', 0)
            end_frame = window.get('end_frame', 0)
            
            # 윈도우 내 각 프레임의 어노테이션 추출
            persons_data = annotation['annotation']
            keypoints_data = annotation.get('keypoints', [])
            
            for frame_offset in range(end_frame - start_frame + 1):
                frame_idx = start_frame + frame_offset
                
                # 해당 프레임의 person 데이터 구성
                frame_persons = []
                
                for person_key, person_data in persons_data.items():
                    if not person_key.startswith('person_'):
                        continue
                    
                    # 키포인트 시퀀스에서 현재 프레임 데이터 추출
                    keypoints_seq = person_data.get('keypoints_sequence', [])
                    bbox_seq = person_data.get('bbox_sequence', [])
                    score_seq = person_data.get('score_sequence', [])
                    
                    if frame_offset < len(keypoints_seq):
                        person_frame_data = {
                            'keypoints': keypoints_seq[frame_offset],
                            'bbox': bbox_seq[frame_offset] if frame_offset < len(bbox_seq) else [],
                            'score': score_seq[frame_offset] if frame_offset < len(score_seq) else 0.0,
                            'track_id': person_data.get('track_id'),
                            'composite_score': person_data.get('composite_score', 0.0),
                            'rank': person_data.get('rank', 999)
                        }
                        frame_persons.append(person_frame_data)
                
                # 프레임 어노테이션 생성
                frame_annotation = {
                    'frame_idx': frame_idx,
                    'timestamp': frame_idx / 30.0,  # 30fps 가정
                    'persons': frame_persons,
                    'window_info': {
                        'window_idx': window.get('window_idx', 0),
                        'start_frame': start_frame,
                        'end_frame': end_frame,
                        'composite_score': window.get('composite_score', 0.0)
                    }
                }
                
                frame_annotations.append(frame_annotation)
        
        # 프레임 인덱스로 정렬
        frame_annotations.sort(key=lambda x: x['frame_idx'])
        
        # 중복 프레임 제거 (같은 프레임이 여러 윈도우에 포함될 수 있음)
        unique_annotations = {}
        for ann in frame_annotations:
            frame_idx = ann['frame_idx']
            if frame_idx not in unique_annotations:
                unique_annotations[frame_idx] = ann
            else:
                # 더 높은 점수의 윈도우 정보 사용
                existing_score = unique_annotations[frame_idx]['window_info']['composite_score']
                new_score = ann['window_info']['composite_score']
                if new_score > existing_score:
                    unique_annotations[frame_idx] = ann
        
        return list(unique_annotations.values())
    
    def _draw_stage2_annotation_on_frame(self, frame: np.ndarray, annotation: Dict[str, Any], 
                                       stage2_data: Dict, frame_idx: int,
                                       show_windows: bool, show_scores: bool) -> np.ndarray:
        """Stage 2 어노테이션을 프레임에 그리기"""
        vis_frame = frame.copy()
        
        # FramePoses 객체 생성 (기존 PoseVisualizer 사용)
        persons = []
        for person_data in annotation['persons']:
            keypoints = person_data.get('keypoints', [])
            if isinstance(keypoints, list) and keypoints:
                keypoints = np.array(keypoints)
            else:
                keypoints = np.array([])
            
            person = PersonPose(
                bbox=person_data.get('bbox', []),
                keypoints=keypoints,
                score=person_data.get('score', 0.0),
                track_id=person_data.get('track_id')
            )
            persons.append(person)
        
        frame_poses = FramePoses(
            frame_idx=frame_idx,
            timestamp=annotation.get('timestamp', 0.0),
            persons=persons
        )
        
        # 기본 포즈 시각화
        vis_frame = self.pose_visualizer.visualize_frame(vis_frame, frame_poses)
        
        # 추가 정보 표시
        if show_scores:
            self._draw_composite_scores(vis_frame, annotation['persons'])
        
        if show_windows:
            self._draw_window_info(vis_frame, annotation.get('window_info', {}))
        
        return vis_frame
    
    def _draw_composite_scores(self, frame: np.ndarray, persons: List[Dict]):
        """복합점수 표시"""
        for i, person in enumerate(persons):
            bbox = person.get('bbox', [])
            composite_score = person.get('composite_score', 0.0)
            rank = person.get('rank', 999)
            
            if len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[:4]
                
                # 점수 텍스트
                score_text = f"Rank:{rank} Score:{composite_score:.3f}"
                
                # 텍스트 배경
                text_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                cv2.rectangle(frame, (int(x1), int(y1) - text_size[1] - 10), 
                             (int(x1) + text_size[0], int(y1)), (0, 0, 0), -1)
                
                # 텍스트
                cv2.putText(frame, score_text, (int(x1), int(y1) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    
    def _draw_window_info(self, frame: np.ndarray, window_info: Dict):
        """윈도우 정보 표시"""
        if not window_info:
            return
        
        window_idx = window_info.get('window_idx', 0)
        start_frame = window_info.get('start_frame', 0)
        end_frame = window_info.get('end_frame', 0)
        composite_score = window_info.get('composite_score', 0.0)
        
        # 윈도우 정보 텍스트
        window_text = f"Window {window_idx}: [{start_frame}-{end_frame}] Score:{composite_score:.3f}"
        
        # 상단에 반투명 배경으로 표시
        text_size = cv2.getTextSize(window_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (text_size[0] + 20, text_size[1] + 20), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 텍스트
        cv2.putText(frame, window_text, (15, text_size[1] + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def _draw_stage2_info(self, frame: np.ndarray, stage2_data: Dict, current_frame: int, total_frames: int):
        """Stage 2 정보 패널 표시"""
        height = frame.shape[0]
        
        # 비디오 정보
        info_lines = [
            f"Video: {stage2_data.get('video_name', 'Unknown')}",
            f"Label: {stage2_data.get('label_folder', 'Unknown')}",
            f"Frame: {current_frame}/{total_frames}",
            f"Total Windows: {stage2_data.get('num_windows', 0)}"
        ]
        
        # 트래킹 설정 정보
        tracking_settings = stage2_data.get('tracking_settings', {})
        settings_lines = [
            f"Window Size: {tracking_settings.get('clip_len', 'N/A')}",
            f"Stride: {tracking_settings.get('training_stride', 'N/A')}",
            f"Quality Thresh: {tracking_settings.get('quality_threshold', 'N/A'):.2f}" if tracking_settings.get('quality_threshold') else "Quality Thresh: N/A"
        ]
        
        # 정보 패널 그리기
        panel_width = 300
        panel_height = len(info_lines + settings_lines) * 25 + 40
        
        # 패널 배경
        overlay = frame.copy()
        cv2.rectangle(overlay, (frame.shape[1] - panel_width - 10, height - panel_height - 10),
                     (frame.shape[1] - 10, height - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        cv2.rectangle(frame, (frame.shape[1] - panel_width - 10, height - panel_height - 10),
                     (frame.shape[1] - 10, height - 10), (255, 255, 255), 1)
        
        # 텍스트 그리기
        y_offset = height - panel_height + 15
        
        for line in info_lines:
            cv2.putText(frame, line, (frame.shape[1] - panel_width + 5, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
        
        # 구분선
        cv2.line(frame, (frame.shape[1] - panel_width + 5, y_offset - 10),
                (frame.shape[1] - 15, y_offset - 10), (128, 128, 128), 1)
        
        for line in settings_lines:
            cv2.putText(frame, line, (frame.shape[1] - panel_width + 5, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 25
        
        # 조작 가이드
        guide_lines = [
            "Controls:",
            "Q: Quit",
            "Space: Pause",
            "S: Toggle Scores", 
            "W: Toggle Windows"
        ]
        
        guide_y = 50
        for line in guide_lines:
            cv2.putText(frame, line, (frame.shape[1] - panel_width + 5, guide_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            guide_y += 20
    
    def create_stage2_summary(self, pkl_path: Union[str, Path], 
                             output_size: Tuple[int, int] = (1400, 900)) -> np.ndarray:
        """Stage 2 pkl 파일 요약 이미지 생성
        
        Args:
            pkl_path: Stage 2 pkl 파일 경로
            output_size: 출력 크기
            
        Returns:
            요약 이미지
        """
        try:
            # pkl 파일 로드
            with open(pkl_path, 'rb') as f:
                stage2_data = pickle.load(f)
            
            width, height = output_size
            summary_image = np.zeros((height, width, 3), dtype=np.uint8)
            
            # 제목
            video_name = stage2_data.get('video_name', 'Unknown')
            title = f"Stage 2 Summary - {video_name}"
            title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
            cv2.putText(summary_image, title, 
                       ((width - title_size[0]) // 2, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            
            # 비디오 정보
            info_text = [
                f"Video: {video_name}",
                f"Label: {stage2_data.get('label_folder', 'Unknown')}",
                f"Dataset: {stage2_data.get('dataset_name', 'Unknown')}",
                f"Total Frames: {stage2_data.get('total_frames', 0)}",
                f"Total Windows: {stage2_data.get('num_windows', 0)}"
            ]
            
            y_offset = 100
            for text in info_text:
                cv2.putText(summary_image, text, (50, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
                y_offset += 30
            
            # 트래킹 설정 정보
            tracking_settings = stage2_data.get('tracking_settings', {})
            settings_text = [
                f"Window Size: {tracking_settings.get('clip_len', 'N/A')}",
                f"Training Stride: {tracking_settings.get('training_stride', 'N/A')}",
                f"Track High Thresh: {tracking_settings.get('track_high_thresh', 'N/A')}",
                f"Track Low Thresh: {tracking_settings.get('track_low_thresh', 'N/A')}",
                f"Quality Threshold: {tracking_settings.get('quality_threshold', 'N/A')}",
                f"Min Track Length: {tracking_settings.get('min_track_length', 'N/A')}"
            ]
            
            y_offset = 100
            for text in settings_text:
                cv2.putText(summary_image, text, (width // 2 + 50, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
                y_offset += 30
            
            # 윈도우 복합점수 분포 (상위 10개)
            windows = stage2_data.get('windows', [])
            if windows:
                y_offset += 50
                cv2.putText(summary_image, "Top 10 Windows (by Composite Score):", (50, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
                y_offset += 40
                
                # 복합점수로 정렬 (이미 정렬되어 있지만 확인)
                sorted_windows = sorted(windows, key=lambda x: x.get('composite_score', 0.0), reverse=True)
                
                for i, window in enumerate(sorted_windows[:10]):
                    window_text = (f"Window {window.get('window_idx', i)}: "
                                 f"[{window.get('start_frame', 0)}-{window.get('end_frame', 0)}] "
                                 f"Score: {window.get('composite_score', 0.0):.3f}")
                    
                    cv2.putText(summary_image, window_text, (70, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 255, 150), 1)
                    y_offset += 25
                
                # 점수 분포 히스토그램 (간단한 버전)
                scores = [w.get('composite_score', 0.0) for w in windows]
                if scores:
                    graph_y = height - 200
                    graph_height = 150
                    graph_width = width - 100
                    
                    max_score = max(scores)
                    min_score = min(scores)
                    
                    # 그래프 제목
                    cv2.putText(summary_image, f"Composite Score Distribution (Range: {min_score:.3f} - {max_score:.3f})", 
                               (50, graph_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
                    
                    # 간단한 점 그래프
                    for i, score in enumerate(scores):
                        if max_score > min_score:
                            x = int(50 + (i * graph_width) / len(scores))
                            y = int(graph_y - ((score - min_score) * graph_height) / (max_score - min_score))
                        else:
                            x = int(50 + (i * graph_width) / len(scores))
                            y = graph_y
                        
                        cv2.circle(summary_image, (x, y), 2, (0, 255, 0), -1)
                    
                    # 축 그리기
                    cv2.line(summary_image, (50, graph_y), (50 + graph_width, graph_y), (255, 255, 255), 1)
                    cv2.line(summary_image, (50, graph_y), (50, graph_y - graph_height), (255, 255, 255), 1)
                    
                    # 축 라벨
                    cv2.putText(summary_image, f"{min_score:.3f}", (45, graph_y + 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(summary_image, f"{max_score:.3f}", (30, graph_y - graph_height + 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return summary_image
            
        except Exception as e:
            logging.error(f"Error creating Stage 2 summary: {str(e)}")
            # 에러 이미지 생성
            width, height = output_size
            error_image = np.zeros((height, width, 3), dtype=np.uint8)
            error_text = f"Error loading Stage 2 data: {str(e)}"
            cv2.putText(error_image, error_text, (50, height // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return error_image