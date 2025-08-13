#!/usr/bin/env python3
"""
Real-time CCTV Fight Detection Runner
실시간 CCTV 폭력 탐지 시스템 실행 스크립트

Usage:
    python run_realtime_detection.py --source rtsp://... --preset balanced
    python run_realtime_detection.py --source 0 --preset high_accuracy  
    python run_realtime_detection.py --source video.mp4 --device cuda:1
"""

import os
import sys
import argparse
import signal
import time
import json
from typing import Dict, Any, Optional, List

# 프로젝트 루트를 Python path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.realtime_fight_detector import RealtimeFightDetector, RealtimeEventLogger, DetectionEvent
from configs.realtime_config import RealtimeConfig, get_config, PRESET_CONFIGS


class RealtimeDetectionRunner:
    """실시간 탐지 실행 관리자"""
    
    def __init__(self, args):
        self.args = args
        self.detector = None
        self.running = False
        
        # 신호 핸들러 설정
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """신호 처리 (Ctrl+C 등)"""
        print(f"\nReceived signal {signum}. Shutting down gracefully...")
        self.running = False
        if self.detector:
            self.detector.stop_detection()
        sys.exit(0)
    
    def setup_config(self) -> Dict[str, Any]:
        """설정 구성"""
        config_manager = RealtimeConfig()
        
        # 소스별 기본 설정 생성
        if self.args.source.startswith('rtsp://'):
            config = config_manager.get_rtsp_config(self.args.source)
        elif self.args.source.isdigit():
            config = config_manager.get_webcam_config(int(self.args.source))
        else:
            config = config_manager.get_video_file_config(self.args.source)
        
        # 프리셋 적용
        if self.args.preset and self.args.preset in PRESET_CONFIGS:
            config.update(PRESET_CONFIGS[self.args.preset])
            print(f"Applied preset: {self.args.preset}")
        
        # 커맨드 라인 오버라이드 적용
        overrides = {}
        
        if self.args.device:
            overrides['device'] = self.args.device
        
        if self.args.threshold:
            overrides['classification_threshold'] = self.args.threshold
        
        if self.args.clip_len:
            overrides['clip_len'] = self.args.clip_len
        
        if self.args.stride:
            overrides['inference_stride'] = self.args.stride
        
        if self.args.fps:
            overrides['stream_config'] = config.get('stream_config', {})
            overrides['stream_config']['target_fps'] = self.args.fps
        
        if self.args.output_dir:
            overrides['output_config'] = config.get('output_config', {})
            overrides['output_config']['output_dir'] = self.args.output_dir
        
        if overrides:
            config.update(overrides)
            print(f"Applied overrides: {list(overrides.keys())}")
        
        # 설정 검증
        errors = config_manager.validate_config(config)
        if errors:
            print("Configuration errors:")
            for error in errors:
                print(f"  - {error}")
            sys.exit(1)
        
        return config
    
    def setup_event_handlers(self, config: Dict[str, Any]) -> List[Any]:
        """이벤트 핸들러 설정"""
        handlers = []
        
        # 파일 로깅
        if config['event_config']['enable_file_logging']:
            log_file = config['event_config']['log_file_path']
            if not os.path.isabs(log_file):
                output_dir = config['output_config']['output_dir']
                os.makedirs(output_dir, exist_ok=True)
                log_file = os.path.join(output_dir, log_file)
            
            logger = RealtimeEventLogger(log_file)
            handlers.append(logger)
            print(f"Event logging enabled: {log_file}")
        
        # 콘솔 알림 핸들러
        if config['event_config']['enable_console_alerts']:
            def console_handler(event: DetectionEvent):
                if event.is_fight:
                    print(f"\n🚨 FIGHT DETECTED! (ID: {event.event_id[:8]}...)")
                    print(f"   Confidence: {event.confidence:.3f}")
                    print(f"   Persons: {event.event_summary['totalPersons']}")
                    print(f"   Time: {event.timestamp}")
                elif event.confidence > config['event_config']['min_confidence_for_alert']:
                    print(f"Normal activity (confidence: {event.confidence:.3f})")
            
            handlers.append(console_handler)
            print("Console alerts enabled")
        
        # 웹훅 핸들러 (선택사항)
        if config['event_config']['enable_webhook_alerts'] and config['event_config']['webhook_url']:
            def webhook_handler(event: DetectionEvent):
                try:
                    import requests
                    if event.is_fight:
                        payload = {
                            'event_id': event.event_id,
                            'timestamp': event.timestamp,
                            'is_fight': event.is_fight,
                            'confidence': event.confidence,
                            'source': str(config['source'])
                        }
                        
                        response = requests.post(
                            config['event_config']['webhook_url'],
                            json=payload,
                            timeout=config['event_config']['webhook_timeout']
                        )
                        
                        if response.status_code == 200:
                            print(f"Webhook notification sent for event {event.event_id[:8]}...")
                        
                except Exception as e:
                    print(f"Webhook error: {e}")
            
            handlers.append(webhook_handler)
            print(f"Webhook alerts enabled: {config['event_config']['webhook_url']}")
        
        return handlers
    
    def display_config_summary(self, config: Dict[str, Any]):
        """설정 요약 출력"""
        print("\n" + "="*60)
        print("REALTIME FIGHT DETECTION SYSTEM")
        print("="*60)
        print(f"Source: {config['source']}")
        print(f"Device: {config['device']}")
        print(f"Classification threshold: {config['classification_threshold']}")
        print(f"Window size: {config['clip_len']} frames")
        print(f"Inference stride: {config['inference_stride']} frames")
        print(f"Target FPS: {config['stream_config']['target_fps']}")
        
        if self.args.preset:
            print(f"Preset: {self.args.preset}")
        
        print("="*60)
    
    def run_interactive_mode(self):
        """대화형 모드 실행"""
        print("\nEntering interactive mode...")
        print("Commands:")
        print("  's' - Show statistics")
        print("  'q' - Quit")
        print("  'h' - Show help")
        
        while self.running:
            try:
                cmd = input("\n> ").strip().lower()
                
                if cmd == 'q':
                    self.running = False
                    break
                elif cmd == 's':
                    self.show_statistics()
                elif cmd == 'h':
                    print("Available commands:")
                    print("  's' - Show current statistics")
                    print("  'q' - Quit application")
                    print("  'h' - Show this help")
                elif cmd == '':
                    continue
                else:
                    print(f"Unknown command: {cmd}")
                    
            except (EOFError, KeyboardInterrupt):
                break
    
    def show_statistics(self):
        """통계 정보 표시"""
        if not self.detector:
            return
        
        stats = self.detector.get_statistics()
        
        print("\n" + "-"*50)
        print("CURRENT STATISTICS")
        print("-"*50)
        
        detector_stats = stats['detector_stats']
        print(f"Frames processed: {detector_stats['frames_processed']}")
        print(f"Windows processed: {detector_stats['windows_processed']}")
        print(f"Fight detections: {detector_stats['fight_detections']}")
        print(f"Average FPS: {detector_stats['avg_fps']:.1f}")
        
        stream_stats = stats['stream_stats']
        print(f"Stream FPS: {stream_stats.get('current_fps', 0):.1f}")
        print(f"Stream status: {'Running' if stream_stats.get('is_running', False) else 'Stopped'}")
        
        window_stats = stats['window_stats']
        print(f"Current frame: {window_stats['current_frame_count']}")
        print(f"Window count: {window_stats['window_count']}")
        print(f"Buffer size: {window_stats['buffer_size']}")
        
        print("-"*50)
    
    def run(self):
        """메인 실행 함수"""
        print("Starting Real-time CCTV Fight Detection System...")
        
        # 설정 구성
        config = self.setup_config()
        self.display_config_summary(config)
        
        # 이벤트 핸들러 설정
        handlers = self.setup_event_handlers(config)
        
        try:
            # 탐지기 초기화
            self.detector = RealtimeFightDetector(config)
            
            # 이벤트 핸들러 등록
            for handler in handlers:
                self.detector.add_event_callback(handler)
            
            # 탐지 시작
            self.detector.start_detection()
            self.running = True
            
            print("\n✅ System started successfully!")
            print("Press Ctrl+C to stop or use interactive commands...")
            
            if self.args.interactive:
                self.run_interactive_mode()
            else:
                # 상태 모니터링 모드
                last_stats_time = time.time()
                
                while self.running:
                    current_time = time.time()
                    
                    if current_time - last_stats_time >= 10:  # 10초마다 상태 출력
                        stats = self.detector.get_statistics()
                        detector_stats = stats['detector_stats']
                        
                        print(f"\r[{time.strftime('%H:%M:%S')}] "
                              f"Frames: {detector_stats['frames_processed']}, "
                              f"FPS: {detector_stats['avg_fps']:.1f}, "
                              f"Fights: {detector_stats['fight_detections']}", 
                              end='', flush=True)
                        
                        last_stats_time = current_time
                    
                    time.sleep(0.1)
        
        except KeyboardInterrupt:
            print("\nReceived interrupt signal...")
        
        except Exception as e:
            print(f"\nError occurred: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # 정리 작업
            if self.detector:
                print("\nShutting down detector...")
                self.detector.stop_detection()
            
            print("System shutdown complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Real-time CCTV Fight Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # RTSP 카메라로부터 탐지 (고정확도 모드)
  python run_realtime_detection.py --source rtsp://admin:password@192.168.1.100:554/stream --preset high_accuracy
  
  # 웹캠으로부터 탐지 (균형 모드)
  python run_realtime_detection.py --source 0 --preset balanced
  
  # 비디오 파일에서 탐지 (고속 모드)
  python run_realtime_detection.py --source video.mp4 --preset high_speed
  
  # 커스텀 설정
  python run_realtime_detection.py --source 0 --threshold 0.7 --device cuda:1 --fps 20

Available presets: high_accuracy, high_speed, balanced, debug
        """
    )
    
    # 필수 인자
    parser.add_argument('--source', required=True,
                       help='Input source (RTSP URL, camera index, or video file)')
    
    # 선택적 인자
    parser.add_argument('--preset', choices=list(PRESET_CONFIGS.keys()),
                       default='balanced',
                       help='Configuration preset (default: balanced)')
    
    parser.add_argument('--device', 
                       help='Device to use (cuda:0, cuda:1, cpu)')
    
    parser.add_argument('--threshold', type=float,
                       help='Classification threshold (0.0-1.0)')
    
    parser.add_argument('--clip-len', type=int,
                       help='Window clip length (frames)')
    
    parser.add_argument('--stride', type=int,
                       help='Inference stride (frames)')
    
    parser.add_argument('--fps', type=int,
                       help='Target FPS for stream processing')
    
    parser.add_argument('--output-dir',
                       help='Output directory for logs and results')
    
    parser.add_argument('--interactive', action='store_true',
                       help='Enable interactive mode with commands')
    
    parser.add_argument('--version', action='version', version='1.0.0')
    
    args = parser.parse_args()
    
    # 실행
    runner = RealtimeDetectionRunner(args)
    runner.run()


if __name__ == "__main__":
    main()