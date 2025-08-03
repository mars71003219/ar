#!/usr/bin/env python3
"""
포즈 처리 파이프라인 에러 로깅 시스템
"""

import os
import json
import traceback
import glob
import re
from datetime import datetime

class ProcessingErrorLogger:
    """처리 과정에서 발생하는 에러들을 로그로 기록하는 클래스"""
    
    def __init__(self, output_dir: str, dataset_name: str = "unknown", resume: bool = False):
        """
        Args:
            output_dir: 로그 파일을 저장할 출력 디렉토리
            dataset_name: 데이터셋 이름
            resume: 이전 로그에 이어쓸지 여부
        """
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 로그 파일 경로들 - dataset_name 폴더 내부에 생성
        dataset_output_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        self.error_log_file = os.path.join(dataset_output_dir, f"processing_errors_{self.session_id}.log")
        self.failed_videos_file = os.path.join(dataset_output_dir, f"failed_videos_{self.session_id}.json")
        self.summary_file = os.path.join(dataset_output_dir, f"processing_summary_{self.session_id}.json")
        
        self.resuming = resume
        if self.resuming:
            self._find_and_set_latest_logs()

        # 통계 정보
        self.stats = {
            'total_videos': 0,
            'successful_videos': 0,
            'failed_videos': 0,
            'failed_windows': 0,
            'error_categories': {},
            'start_time': datetime.now().isoformat(),
            'end_time': None
        }
        
        # 실패한 비디오 목록
        self.failed_videos = []
        
        if self.resuming:
            self._load_previous_state()

        # 로그 파일 초기화
        self._initialize_log_files()

    def _find_and_set_latest_logs(self):
        """가장 최신 로그 파일을 찾아 세션 정보를 설정합니다."""
        dataset_output_dir = os.path.join(self.output_dir, self.dataset_name)
        log_files = glob.glob(os.path.join(dataset_output_dir, 'processing_errors_*.log'))
        if not log_files:
            self.resuming = False
            return

        latest_log_file = max(log_files, key=os.path.getmtime)
        self.error_log_file = latest_log_file
        
        match = re.search(r'processing_errors_(\d{8}_\d{6})\.log', os.path.basename(latest_log_file))
        if match:
            self.session_id = match.group(1)
            self.failed_videos_file = os.path.join(dataset_output_dir, f"failed_videos_{self.session_id}.json")
            self.summary_file = os.path.join(dataset_output_dir, f"processing_summary_{self.session_id}.json")
        else:
            # Fallback if pattern doesn't match
            self.resuming = False

    def _load_previous_state(self):
        """이전 세션의 통계와 실패 목록을 불러옵니다."""
        if os.path.exists(self.summary_file):
            try:
                with open(self.summary_file, 'r', encoding='utf-8') as f:
                    previous_stats = json.load(f)
                
                # Resume 시에는 비디오 카운트를 누적하지 않고 새로 시작
                # 단, error_categories는 유지
                if 'error_categories' in previous_stats:
                    self.stats['error_categories'] = previous_stats['error_categories'].copy()
                
                # 원래 시작 시간 유지
                if 'start_time' in previous_stats:
                    self.stats['start_time'] = previous_stats['start_time']
                
                # 비디오 카운트는 0부터 시작 (resume에서 새로 처리할 비디오들만 카운트)
                self.stats['total_videos'] = 0
                self.stats['successful_videos'] = 0
                self.stats['failed_videos'] = 0
                self.stats['failed_windows'] = previous_stats.get('failed_windows', 0)
                self.stats['end_time'] = None
                
            except (json.JSONDecodeError, FileNotFoundError):
                self.stats['start_time'] = datetime.now().isoformat() # Start new if corrupt

        if os.path.exists(self.failed_videos_file):
            try:
                with open(self.failed_videos_file, 'r', encoding='utf-8') as f:
                    self.failed_videos = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                self.failed_videos = [] # Start new if corrupt
    
    def _initialize_log_files(self):
        """로그 파일들 초기화"""
        try:
            # dataset_name 폴더는 이미 __init__에서 생성되었으므로 여기서는 생성하지 않음
            dataset_output_dir = os.path.join(self.output_dir, self.dataset_name)
            os.makedirs(dataset_output_dir, exist_ok=True)
            
            # 에러 로그 파일 초기화
            mode = 'a' if self.resuming and os.path.exists(self.error_log_file) else 'w'
            with open(self.error_log_file, mode, encoding='utf-8') as f:
                if mode == 'w':
                    f.write(f"Processing Error Log - {self.dataset_name}\n")
                    f.write(f"Session ID: {self.session_id}\n")
                    f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("=" * 80 + "\n\n")
                else:
                    f.write("\n" + "=" * 80 + "\n")
                    f.write(f"RESUMING LOGGING at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("=" * 80 + "\n\n")
            
            # 에러 로깅 초기화 완료
            
        except Exception as e:
            # 로깅 초기화 실패 - 계속 진행
            pass
    
    def log_info(self, message: str):
        """일반 정보 메시지를 로그에 기록합니다."""
        self._append_to_log(f"[INFO] {message.strip()}")

    def log_video_start(self, video_path: str):
        """비디오 처리 시작 로그"""
        self.stats['total_videos'] += 1
        self._append_to_log(f"[START] Processing: {os.path.basename(video_path)}")
    
    def log_video_success(self, video_path: str, windows_count: int):
        """비디오 처리 성공 로그"""
        self.stats['successful_videos'] += 1
        self._append_to_log(f"[SUCCESS] {os.path.basename(video_path)} - {windows_count} windows generated")
    
    def log_video_failure(self, video_path: str, failure_analysis: dict = None, error_type: str = None, error_message: str = None, full_traceback: str = None):
        """비디오 처리 실패 로그 - 상세 실패 분석 버전"""
        self.stats['failed_videos'] += 1
        
        # failure_analysis가 있으면 그것을 우선 사용 (새로운 상세 분석)
        if failure_analysis and isinstance(failure_analysis, dict):
            failure_stage = failure_analysis.get('failure_stage', 'UNKNOWN')
            root_cause = failure_analysis.get('root_cause', 'Unknown error')
            detailed_info = failure_analysis.get('detailed_info', {})
            
            # 에러 카테고리 통계 (failure_stage 기준으로)
            if failure_stage not in self.stats['error_categories']:
                self.stats['error_categories'][failure_stage] = 0
            self.stats['error_categories'][failure_stage] += 1
            
            # 실패 정보 기록 (상세 분석 버전)
            failure_info = {
                'video_path': video_path,
                'video_name': os.path.basename(video_path),
                'failure_stage': failure_stage,
                'root_cause': root_cause,
                'suggested_solution': failure_analysis.get('suggested_solution', 'Review error logs'),
                'detailed_info': detailed_info,
                'error_type': failure_analysis.get('error_type', 'Unknown'),
                'error_message': failure_analysis.get('error_message', 'No message'),
                'timestamp': failure_analysis.get('timestamp') or datetime.now().isoformat(),
                'full_traceback': failure_analysis.get('full_traceback')
            }
            
            self.failed_videos.append(failure_info)
            
            # 로그 파일에 상세 정보 기록
            self._append_to_log(f"[FAILED] {os.path.basename(video_path)}")
            self._append_to_log(f"  Failure Stage: {failure_stage}")
            self._append_to_log(f"  Root Cause: {root_cause}")
            self._append_to_log(f"  Suggested Solution: {failure_analysis.get('suggested_solution', 'Review error logs')}")
            
            # 상세 정보 로깅
            if detailed_info:
                self._append_to_log(f"  Detailed Analysis:")
                for key, value in detailed_info.items():
                    if key not in ['analysis_error']:  # 분석 에러는 별도 처리
                        self._append_to_log(f"    {key}: {value}")
            
            self._append_to_log(f"  Error Type: {failure_info['error_type']}")
            self._append_to_log(f"  Error Message: {failure_info['error_message']}")
            
            # 트레이스백은 핵심 부분만 표시
            if failure_info.get('full_traceback'):
                lines = failure_info['full_traceback'].split('\n')
                key_lines = [line for line in lines if any(keyword in line.lower() for keyword in 
                            ['rtmo', 'pose', 'track', 'window', 'annotation', 'error:', 'exception:'])][:5]
                if key_lines:
                    self._append_to_log(f"  Key Error Lines:")
                    for line in key_lines:
                        if line.strip():
                            self._append_to_log(f"    {line.strip()}")
        else:
            # 기존 방식 (하위 호환성)
            if error_type and error_type not in self.stats['error_categories']:
                self.stats['error_categories'][error_type] = 0
                self.stats['error_categories'][error_type] += 1
            
            failure_info = {
                'video_path': video_path,
                'video_name': os.path.basename(video_path),
                'error_type': error_type or 'UNKNOWN',
                'error_message': error_message or 'No error message',
                'timestamp': datetime.now().isoformat(),
                'full_traceback': full_traceback
            }
            
            self.failed_videos.append(failure_info)
            
            # 로그 파일에 기록
            self._append_to_log(f"[FAILED] {os.path.basename(video_path)}")
            self._append_to_log(f"  Error Type: {error_type or 'UNKNOWN'}")
            self._append_to_log(f"  Error Message: {error_message or 'No error message'}")
            if full_traceback:
                self._append_to_log(f"  Full Traceback:")
                for line in full_traceback.split('\n'):
                    if line.strip():
                        self._append_to_log(f"    {line}")
        
        self._append_to_log("")
    
    def log_window_failure(self, video_path: str, window_idx: int, error_message: str):
        """윈도우 처리 실패 로그"""
        self.stats['failed_windows'] += 1
        self._append_to_log(f"[WINDOW_FAILED] {os.path.basename(video_path)} - Window {window_idx}: {error_message}")
    
    def log_general_error(self, component: str, error_message: str, full_traceback: str = None):
        """일반적인 에러 로그"""
        self._append_to_log(f"[ERROR] {component}: {error_message}")
        if full_traceback:
            self._append_to_log(f"  Traceback:")
            for line in full_traceback.split('\n'):
                if line.strip():
                    self._append_to_log(f"    {line}")
        self._append_to_log("")
    
    def _append_to_log(self, message: str):
        """로그 파일에 메시지 추가"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.error_log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] {message}\n")
        except Exception as e:
            pass  # 로그 쓰기 실패 - 계속 진행
    
    def finalize_logging(self):
        """로깅 세션 종료 및 요약 생성"""
        self.stats['end_time'] = datetime.now().isoformat()
        
        # 실패한 비디오 목록 저장
        try:
            with open(self.failed_videos_file, 'w', encoding='utf-8') as f:
                json.dump(self.failed_videos, f, indent=2, ensure_ascii=False)
        except Exception as e:
            pass  # 실패 비디오 목록 저장 실패
        
        # 처리 요약 저장
        try:
            with open(self.summary_file, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, indent=2, ensure_ascii=False)
        except Exception as e:
            pass  # 처리 요약 저장 실패
        
        # 로그 파일에 요약 추가
        self._append_to_log("")
        self._append_to_log("=" * 80)
        self._append_to_log("PROCESSING SUMMARY")
        self._append_to_log("=" * 80)
        self._append_to_log(f"Total Videos: {self.stats['total_videos']}")
        self._append_to_log(f"Successful: {self.stats['successful_videos']}")
        self._append_to_log(f"Failed: {self.stats['failed_videos']}")
        self._append_to_log(f"Failed Windows: {self.stats['failed_windows']}")
        self._append_to_log(f"Success Rate: {self.get_success_rate():.1f}%")
        
        if self.stats['error_categories']:
            self._append_to_log("\nError Categories:")
            for error_type, count in self.stats['error_categories'].items():
                self._append_to_log(f"  {error_type}: {count}")
        
        self._append_to_log(f"\nSession completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 콘솔에 요약 출력
        self.print_summary()
    
    def get_success_rate(self) -> float:
        """성공률 계산"""
        if self.stats['total_videos'] == 0:
            return 0.0
        return (self.stats['successful_videos'] / self.stats['total_videos']) * 100
    
    def print_summary(self):
        """콘솔에 요약 정보 출력"""
        print("\n" + "=" * 70)
        print(" PROCESSING SUMMARY")
        print("=" * 70)
        print(f"Total Videos: {self.stats['total_videos']:,}")
        print(f"Successful: {self.stats['successful_videos']:,}")
        print(f"Failed: {self.stats['failed_videos']:,}")
        print(f"Failed Windows: {self.stats['failed_windows']:,}")
        print(f"Success Rate: {self.get_success_rate():.1f}%")
        
        if self.stats['error_categories']:
            print("\nError Categories:")
            for error_type, count in sorted(self.stats['error_categories'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {error_type}: {count}")
        
        if self.stats['failed_videos'] > 0:
            print(f"\nDetailed error logs saved to:")
            print(f"  {self.error_log_file}")
            print(f"  {self.failed_videos_file}")
            print(f"  {self.summary_file}")

def capture_exception_info() -> tuple:
    """현재 예외 정보를 캡처하여 반환"""
    import sys
    exc_type, exc_value, exc_traceback = sys.exc_info()
    
    if exc_type is None:
        return "Unknown", "No exception information available", ""
    
    error_type = exc_type.__name__
    error_message = str(exc_value)
    full_traceback = traceback.format_exception(exc_type, exc_value, exc_traceback)
    full_traceback_str = ''.join(full_traceback)
    
    return error_type, error_message, full_traceback_str
