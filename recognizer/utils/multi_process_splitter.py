"""
멀티 프로세스 데이터 분할 및 통합 시스템

여러 메인 프로세스를 실행하여 데이터를 분할 처리하고 결과를 통합
"""

import os
import time
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import math
import shutil
import yaml

logger = logging.getLogger(__name__)


class VideoDataSplitter:
    """비디오 데이터 분할기"""
    
    def __init__(self, input_dir: str, num_splits: int = 4):
        self.input_dir = Path(input_dir)
        self.num_splits = num_splits
        self.video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        
    def get_all_videos(self) -> List[Path]:
        """모든 비디오 파일 목록 반환"""
        video_files = []
        for ext in self.video_extensions:
            video_files.extend(self.input_dir.rglob(f"*{ext}"))
            video_files.extend(self.input_dir.rglob(f"*{ext.upper()}"))
        
        return sorted(video_files)
    
    def split_videos(self) -> List[List[Path]]:
        """비디오를 균등하게 분할"""
        all_videos = self.get_all_videos()
        total_videos = len(all_videos)
        
        if total_videos == 0:
            logger.warning("No video files found")
            return []
        
        # 각 분할의 크기 계산
        chunk_size = math.ceil(total_videos / self.num_splits)
        
        splits = []
        for i in range(self.num_splits):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_videos)
            
            if start_idx < total_videos:
                split_videos = all_videos[start_idx:end_idx]
                splits.append(split_videos)
        
        logger.info(f"Split {total_videos} videos into {len(splits)} groups:")
        for i, split in enumerate(splits):
            logger.info(f"  Split {i}: {len(split)} videos")
        
        return splits
    
    def create_split_directories(self, base_output_dir: str) -> List[str]:
        """분할별 임시 디렉토리 생성"""
        split_dirs = []
        base_path = Path(base_output_dir)
        
        for i in range(self.num_splits):
            split_dir = base_path / f"split_{i}"
            split_dir.mkdir(parents=True, exist_ok=True)
            split_dirs.append(str(split_dir))
        
        return split_dirs


class ConfigGenerator:
    """분할별 설정 파일 생성기"""
    
    def __init__(self, base_config_path: str):
        self.base_config_path = base_config_path
        with open(base_config_path, 'r', encoding='utf-8') as f:
            self.base_config = yaml.safe_load(f)
    
    def create_split_configs(
        self, 
        video_splits: List[List[Path]], 
        split_dirs: List[str]
    ) -> List[str]:
        """분할별 설정 파일 생성"""
        config_paths = []
        
        for i, (videos, split_dir) in enumerate(zip(video_splits, split_dirs)):
            # 임시 비디오 디렉토리 생성
            temp_video_dir = Path(split_dir) / "videos"
            temp_video_dir.mkdir(exist_ok=True)
            
            # 비디오 파일 심볼릭 링크 생성 (원본 폴더 구조 보존)
            for video in videos:
                # 원본 폴더명 보존 (fight, normal 등)
                original_parent = video.parent.name
                target_subdir = temp_video_dir / original_parent
                target_subdir.mkdir(exist_ok=True)
                
                link_path = target_subdir / video.name
                if not link_path.exists():
                    try:
                        os.symlink(str(video.absolute()), str(link_path))
                    except:
                        # 심볼릭 링크 실패시 복사
                        shutil.copy2(str(video), str(link_path))
            
            # 분할별 설정 생성
            split_config = self.base_config.copy()
            split_config['annotation']['input'] = str(temp_video_dir)
            split_config['annotation']['output_dir'] = split_dir

            # 모드를 stage1으로 명시적 설정
            split_config['mode'] = 'annotation.stage1'
            
            # 멀티 프로세스 비활성화 (subprocess에서는 단일 처리만)
            if 'multi_process' in split_config:
                split_config['multi_process']['enabled'] = False
            
            # annotation 하위의 멀티프로세스 설정도 비활성화 (호환성)
            if 'annotation' in split_config and 'multi_process' in split_config['annotation']:
                split_config['annotation']['multi_process']['enabled'] = False
            
            # pipeline_mode 비활성화 및 stage1만 활성화 (무한루프 방지)
            if 'annotation' in split_config:
                split_config['annotation']['pipeline_mode'] = False

                # stage1만 활성화하고 나머지는 비활성화
                if 'stage1' in split_config['annotation']:
                    split_config['annotation']['stage1']['enabled'] = True
                    # stage1 멀티프로세스도 비활성화 (subprocess에서는 단일 처리)
                    if 'multi_process' in split_config['annotation']['stage1']:
                        split_config['annotation']['stage1']['multi_process']['enabled'] = False
                    split_config['annotation']['stage1']['enable_parallel'] = False

                if 'stage2' in split_config['annotation']:
                    split_config['annotation']['stage2']['enabled'] = False
                    split_config['annotation']['stage2']['enable_parallel'] = False

                if 'stage3' in split_config['annotation']:
                    split_config['annotation']['stage3']['enabled'] = False
            
            # 설정 파일 저장
            config_path = Path(split_dir) / f"config_split_{i}.yaml"
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(split_config, f, allow_unicode=True)
            
            config_paths.append(str(config_path))
            logger.info(f"Created config for split {i}: {len(videos)} videos -> {config_path}")
        
        return config_paths


class MultiProcessRunner:
    """멀티 프로세스 실행기"""
    
    def __init__(self, main_script_path: str = "/workspace/recognizer/main.py"):
        self.main_script_path = main_script_path
        self.processes = []
    
    def run_split_processes(self, config_paths: List[str], gpu_assignments: List[int] = None) -> List[subprocess.Popen]:
        """분할별 프로세스 실행"""
        processes = []
        
        if gpu_assignments is None:
            gpu_assignments = [i % 2 for i in range(len(config_paths))]  # GPU 0, 1 순환
        
        for i, config_path in enumerate(config_paths):
            gpu_id = gpu_assignments[i]
            
            # 환경 변수 설정
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            
            # 프로세스 실행
            cmd = [
                'python3', 
                self.main_script_path,
                '--config', config_path
            ]
            
            logger.info(f"Starting split {i} on GPU {gpu_id}: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            processes.append(process)
            time.sleep(2)  # 프로세스 시작 간격
        
        self.processes = processes
        return processes
    
    def monitor_processes(self, processes: List[subprocess.Popen]) -> Dict[int, int]:
        """프로세스 병렬 모니터링 - 안전한 종료 보장"""
        import threading
        import queue
        import signal
        
        results = {}
        result_queue = queue.Queue()
        monitor_threads = []
        
        def monitor_single_process(i, process):
            """단일 프로세스 모니터링 스레드"""
            try:
                logger.info(f"Monitoring split {i} (PID: {process.pid})")
                
                # 실시간 출력 모니터링
                try:
                    for line in process.stdout:
                        if line:  # 빈 라인 제외
                            logger.info(f"Split {i}: {line.strip()}")
                except Exception as e:
                    logger.warning(f"Output monitoring error for split {i}: {e}")
                
                # 프로세스 완료 대기
                return_code = process.wait()
                
                if return_code == 0:
                    logger.info(f"Split {i} completed successfully")
                else:
                    logger.error(f"Split {i} failed with return code {return_code}")
                
                result_queue.put((i, return_code))
                
            except Exception as e:
                logger.error(f"Error monitoring split {i}: {e}")
                result_queue.put((i, -1))
            finally:
                # 프로세스 정리
                try:
                    if process.poll() is None:  # 아직 실행 중이면
                        process.terminate()
                        process.wait(timeout=5)
                except Exception:
                    pass
        
        def cleanup_handler(signum, frame):
            """시그널 핸들러 - 모든 프로세스 정리"""
            logger.warning(f"Received signal {signum}, cleaning up processes...")
            for i, process in enumerate(processes):
                if process.poll() is None:
                    try:
                        logger.info(f"Terminating split {i} (PID: {process.pid})")
                        process.terminate()
                        process.wait(timeout=3)
                    except Exception as e:
                        logger.error(f"Error terminating split {i}: {e}")
                        try:
                            process.kill()
                        except Exception:
                            pass
        
        # 시그널 핸들러 등록
        original_sigint_handler = signal.signal(signal.SIGINT, cleanup_handler)
        original_sigterm_handler = signal.signal(signal.SIGTERM, cleanup_handler)
        
        try:
            # 각 프로세스별로 모니터링 스레드 시작
            for i, process in enumerate(processes):
                thread = threading.Thread(target=monitor_single_process, args=(i, process))
                thread.daemon = True
                thread.start()
                monitor_threads.append(thread)
            
            # 모든 결과 수집 (타임아웃 포함)
            collected_results = 0
            timeout_per_result = 300  # 각 결과당 5분 타임아웃
            
            while collected_results < len(processes):
                try:
                    i, return_code = result_queue.get(timeout=timeout_per_result)
                    results[i] = return_code
                    collected_results += 1
                    logger.info(f"Collected result for split {i}: return_code={return_code}")
                except queue.Empty:
                    logger.error("Timeout waiting for process results")
                    break
            
            # 남은 프로세스들 강제 종료
            for i, process in enumerate(processes):
                if i not in results and process.poll() is None:
                    logger.warning(f"Force terminating split {i}")
                    try:
                        process.terminate()
                        process.wait(timeout=5)
                        results[i] = -1
                    except Exception:
                        try:
                            process.kill()
                            results[i] = -1
                        except Exception:
                            pass
            
            # 모든 스레드 완료 대기 (타임아웃 포함)
            for thread in monitor_threads:
                thread.join(timeout=10)
                if thread.is_alive():
                    logger.warning("Monitor thread did not terminate gracefully")
        
        finally:
            # 시그널 핸들러 복원
            signal.signal(signal.SIGINT, original_sigint_handler)
            signal.signal(signal.SIGTERM, original_sigterm_handler)
        
        return results


class ResultMerger:
    """결과 통합기"""
    
    def __init__(self, final_output_dir: str):
        self.final_output_dir = Path(final_output_dir)
        self.final_output_dir.mkdir(parents=True, exist_ok=True)
    
    def merge_stage_results(self, split_dirs: List[str], stage: str) -> bool:
        """스테이지별 결과 통합 - Stage3는 특별 처리"""
        try:
            if stage == 'stage3':
                return self._merge_stage3_results(split_dirs, str(self.final_output_dir))
            else:
                return self._merge_regular_stage_results(split_dirs, stage)

        except Exception as e:
            logger.error(f"Error merging {stage} results: {e}")
            return False

    def _merge_stage3_results(self, split_dirs: List[str], output_dir: str) -> bool:
        """Stage3 임시 파일들을 통합하여 최종 데이터셋 생성"""
        try:
            from pipelines.dual_service.dual_pipeline import DualServicePipeline
            import yaml
            import pickle
            import random

            # Stage3 데이터셋 디렉토리 찾기
            temp_dirs = []
            for split_dir in split_dirs:
                split_path = Path(split_dir)
                # temp_stage3과 stage3_dataset 디렉토리 모두 확인
                temp_stage3_dirs = list(split_path.rglob("**/temp_stage3"))
                stage3_dataset_dirs = list(split_path.rglob("**/stage3_dataset"))
                temp_dirs.extend([str(d) for d in temp_stage3_dirs])
                temp_dirs.extend([str(d) for d in stage3_dataset_dirs])

            if not temp_dirs:
                logger.warning("No stage3 temp directories found")
                return False

            logger.info(f"Found {len(temp_dirs)} stage3 temp directories")

            # 설정에서 split_ratios 가져오기
            split_ratios = {'train': 0.7, 'val': 0.2, 'test': 0.1}
            try:
                # 첫 번째 split 디렉토리에서 설정 파일 찾기
                first_split = Path(split_dirs[0])
                config_files = list(first_split.rglob("config_split_*.yaml"))
                if config_files:
                    with open(config_files[0], 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                        split_ratios = config.get('annotation', {}).get('stage3', {}).get('split_ratios', split_ratios)
            except Exception as e:
                logger.warning(f"Failed to load split_ratios from config, using defaults: {e}")

            # 모든 임시 파일과 stage3 데이터셋 파일들을 수집하여 통합
            all_temp_files = []
            all_stage3_files = {'train': [], 'val': [], 'test': []}

            for temp_dir in temp_dirs:
                temp_path = Path(temp_dir)
                if temp_path.exists():
                    # 기존 temp 파일들 찾기
                    temp_files = list(temp_path.glob("*_temp.pkl"))
                    all_temp_files.extend(temp_files)
                    if temp_files:
                        logger.info(f"Found {len(temp_files)} temp files in {temp_dir}")

                    # stage3_dataset 내의 train/val/test 파일들 찾기
                    for split_type in ['train', 'val', 'test']:
                        split_files = list(temp_path.rglob(f"**/{split_type}.pkl"))
                        all_stage3_files[split_type].extend(split_files)
                        if split_files:
                            logger.info(f"Found {len(split_files)} {split_type} files in {temp_dir}")

            # temp 파일 또는 stage3 데이터셋 파일이 있는지 확인
            has_temp_files = len(all_temp_files) > 0
            has_stage3_files = any(len(files) > 0 for files in all_stage3_files.values())

            if not has_temp_files and not has_stage3_files:
                logger.warning("No stage3 temp files or dataset files found for merging")
                return False

            # Stage3 데이터셋 파일들이 있는 경우 (새로운 방식)
            if has_stage3_files:
                return self._merge_stage3_dataset_files(all_stage3_files, output_dir, split_ratios)

            # 기존 temp 파일 처리 방식 (하위 호환성)
            if not all_temp_files:
                logger.warning("No stage3 temp files found for merging")
                return False

            logger.info(f"Total stage3 temp files to merge: {len(all_temp_files)}")

            # 모든 데이터를 한 번에 수집
            all_entries = []
            video_count = 0
            label_counts = {}

            for temp_file in all_temp_files:
                try:
                    with open(temp_file, 'rb') as f:
                        temp_data = pickle.load(f)

                    entries = temp_data.get('dataset_entries', [])
                    label = temp_data.get('label', 0)
                    video_name = temp_data.get('video_name', 'unknown')

                    all_entries.extend(entries)
                    video_count += 1

                    label_counts[label] = label_counts.get(label, 0) + len(entries)

                    logger.info(f"Merged {video_name}: {len(entries)} entries (label: {label})")

                except Exception as e:
                    logger.error(f"Error reading temp file {temp_file}: {e}")
                    continue

            if not all_entries:
                logger.warning("No valid entries found in temp files")
                return False

            # 데이터 셔플
            random.shuffle(all_entries)

            # Train/Val/Test 분할
            total = len(all_entries)
            train_end = int(total * split_ratios['train'])
            val_end = int(total * (split_ratios['train'] + split_ratios['val']))

            train_data = all_entries[:train_end]
            val_data = all_entries[train_end:val_end]
            test_data = all_entries[val_end:]

            # 최종 출력 디렉토리 설정
            stage3_output_dir = self.final_output_dir / "stage3_dataset"
            stage3_output_dir.mkdir(parents=True, exist_ok=True)

            # 최종 파일 저장
            train_file = stage3_output_dir / "train.pkl"
            val_file = stage3_output_dir / "val.pkl"
            test_file = stage3_output_dir / "test.pkl"

            with open(train_file, 'wb') as f:
                pickle.dump(train_data, f)

            with open(val_file, 'wb') as f:
                pickle.dump(val_data, f)

            with open(test_file, 'wb') as f:
                pickle.dump(test_data, f)

            # 임시 파일 정리
            for temp_file in all_temp_files:
                try:
                    temp_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to remove temp file {temp_file}: {e}")

            total_merged = total

            logger.info(f"Stage3 merge completed: {total_merged} total entries")
            return total_merged > 0

        except Exception as e:
            logger.error(f"Error in stage3 merge: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _merge_stage3_dataset_files(self, all_stage3_files: Dict, output_dir: str, split_ratios: Dict) -> bool:
        """Stage3 데이터셋 파일들을 병합"""
        try:
            import pickle

            logger.info("Merging Stage3 dataset files...")

            # 최종 출력 디렉토리 생성
            final_output_dir = Path(output_dir) / "stage3_dataset"
            final_output_dir.mkdir(parents=True, exist_ok=True)

            # 각 split별로 데이터 수집 및 병합
            merged_data = {'train': [], 'val': [], 'test': []}
            total_counts = {'train': 0, 'val': 0, 'test': 0}

            for split_type in ['train', 'val', 'test']:
                split_files = all_stage3_files[split_type]
                logger.info(f"Processing {len(split_files)} {split_type} files")

                for split_file in split_files:
                    try:
                        with open(split_file, 'rb') as f:
                            data = pickle.load(f)

                        if isinstance(data, list):
                            merged_data[split_type].extend(data)
                            total_counts[split_type] += len(data)
                            logger.info(f"Merged {split_file.name}: {len(data)} entries")
                        else:
                            logger.warning(f"Unexpected data format in {split_file}: {type(data)}")

                    except Exception as e:
                        logger.error(f"Error processing {split_file}: {e}")
                        continue

                # 병합된 데이터 저장
                if merged_data[split_type]:
                    output_file = final_output_dir / f"{split_type}.pkl"
                    with open(output_file, 'wb') as f:
                        pickle.dump(merged_data[split_type], f)
                    logger.info(f"Saved merged {split_type}.pkl: {len(merged_data[split_type])} entries")

            # 요약 정보 저장
            total_entries = sum(total_counts.values())
            summary = {
                'train_count': total_counts['train'],
                'val_count': total_counts['val'],
                'test_count': total_counts['test'],
                'total_entries': total_entries,
                'split_ratios': split_ratios,
                'output_dir': str(final_output_dir)
            }

            summary_file = final_output_dir / "dataset_summary.json"
            with open(summary_file, 'w') as f:
                import json
                json.dump(summary, f, indent=2)

            logger.info(f"✅ Stage3 dataset merge completed")
            logger.info(f"  - Train: {total_counts['train']} entries")
            logger.info(f"  - Val: {total_counts['val']} entries")
            logger.info(f"  - Test: {total_counts['test']} entries")
            logger.info(f"  - Total: {total_entries} entries")
            logger.info(f"  - Output: {final_output_dir}")

            return True

        except Exception as e:
            logger.error(f"Error in stage3 dataset merge: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _merge_regular_stage_results(self, split_dirs: List[str], stage: str) -> bool:
        """일반 스테이지 (Stage1, Stage2) 결과 통합"""
        try:
            # 최종 출력 디렉토리 구조 생성
            stage_patterns = {
                'stage1': '**/stage1_poses/**/*.pkl',
                'stage2': '**/stage2_tracking/**/*.pkl'
            }

            pattern = stage_patterns.get(stage, f'**/{stage}**/*.pkl')
            merged_count = 0
            duplicate_count = 0

            for split_dir in split_dirs:
                split_path = Path(split_dir)

                # 해당 스테이지 결과 파일들 찾기
                result_files = list(split_path.rglob(pattern))

                for result_file in result_files:
                    # 상대 경로 계산하되 videos/ 부분을 제거
                    try:
                        rel_path = result_file.relative_to(split_path)

                        # videos/ 경로를 제거하고 재구성
                        path_parts = list(rel_path.parts)
                        if path_parts and path_parts[0] == 'videos':
                            # videos/stage1_poses/... -> stage1_poses/...
                            new_rel_path = Path(*path_parts[1:])
                        else:
                            new_rel_path = rel_path

                        final_path = self.final_output_dir / new_rel_path

                        # 중복 파일 처리
                        if final_path.exists():
                            duplicate_count += 1
                            logger.warning(f"Duplicate file found, skipping: {final_path}")
                            continue

                        # 디렉토리 생성
                        final_path.parent.mkdir(parents=True, exist_ok=True)

                        # 파일 이동 (안전한 이동)
                        try:
                            shutil.move(str(result_file), str(final_path))
                            merged_count += 1
                        except Exception as move_error:
                            # 이동 실패시 복사 시도
                            logger.warning(f"Move failed, trying copy: {move_error}")
                            shutil.copy2(str(result_file), str(final_path))
                            os.remove(str(result_file))  # 원본 파일 삭제
                            merged_count += 1

                    except Exception as e:
                        logger.warning(f"Failed to merge {result_file}: {e}")

            if duplicate_count > 0:
                logger.warning(f"Found {duplicate_count} duplicate files for {stage}")

            logger.info(f"Merged {merged_count} files for {stage}")
            return merged_count > 0

        except Exception as e:
            logger.error(f"Error merging {stage} results: {e}")
            return False
    
    def cleanup_split_dirs(self, split_dirs: List[str]):
        """분할 디렉토리 완전 정리"""
        # temp_splits 폴더 전체 삭제
        if split_dirs:
            temp_base = Path(split_dirs[0]).parent
            try:
                if temp_base.name.endswith("_temp_splits"):
                    shutil.rmtree(str(temp_base))
                    logger.info(f"Cleaned up temp_splits directory: {temp_base}")
                    return
            except Exception as e:
                logger.warning(f"Failed to cleanup temp_splits: {e}")
        
        # 개별 분할 디렉토리 삭제
        for split_dir in split_dirs:
            try:
                shutil.rmtree(split_dir)
                logger.info(f"Cleaned up split directory: {split_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {split_dir}: {e}")


class MultiProcessAnnotationManager:
    """멀티 프로세스 어노테이션 관리자"""
    
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        config_path: str,
        num_processes: int = 4,
        gpu_assignments: List[int] = None
    ):
        self.input_dir = input_dir
        self.config_path = config_path
        self.num_processes = num_processes
        self.gpu_assignments = gpu_assignments or [i % 2 for i in range(num_processes)]
        
        # 출력 디렉토리를 입력 폴더명으로 설정
        input_folder_name = Path(input_dir).name
        self.output_dir = str(Path(output_dir) / input_folder_name)
        
        # 컴포넌트 초기화
        self.splitter = VideoDataSplitter(input_dir, num_processes)
        self.config_generator = ConfigGenerator(config_path)
        self.runner = MultiProcessRunner()
        self.merger = ResultMerger(self.output_dir)
    
    def run_full_pipeline(self) -> bool:
        """전체 파이프라인 실행"""
        start_time = time.time()
        
        try:
            logger.info(f"Starting multi-process annotation with {self.num_processes} processes")
            logger.info(f"Input: {self.input_dir}")
            logger.info(f"Output: {self.output_dir}")
            logger.info(f"GPU assignments: {self.gpu_assignments}")
            
            # 1. 비디오 분할
            logger.info("=== Step 1: Splitting videos ===")
            video_splits = self.splitter.split_videos()
            if not video_splits:
                logger.error("No videos to process")
                return False
            
            # 2. 임시 디렉토리 생성 (입력 디렉토리 기반으로)
            input_parent = Path(self.input_dir).parent
            input_folder_name = Path(self.input_dir).name
            temp_base = input_parent / f"{input_folder_name}_temp_splits"
            split_dirs = self.splitter.create_split_directories(str(temp_base))
            
            # 3. 분할별 설정 파일 생성
            logger.info("=== Step 2: Creating split configurations ===")
            config_paths = self.config_generator.create_split_configs(video_splits, split_dirs)
            
            # 4. 멀티 프로세스 실행
            logger.info("=== Step 3: Running split processes ===")
            processes = self.runner.run_split_processes(config_paths, self.gpu_assignments)
            
            # 5. 프로세스 모니터링
            logger.info("=== Step 4: Monitoring processes ===")
            results = self.runner.monitor_processes(processes)
            
            # 6. 결과 통합
            logger.info("=== Step 5: Merging results ===")
            process_success_count = sum(1 for code in results.values() if code == 0)
            logger.info(f"Completed processes: {process_success_count}/{len(processes)}")

            # 각 스테이지별 결과 통합
            stage_results = {}
            for stage in ['stage1', 'stage2', 'stage3']:
                stage_results[stage] = self.merger.merge_stage_results(split_dirs, stage)

            # 7. 정리
            logger.info("=== Step 6: Cleanup ===")
            self.merger.cleanup_split_dirs(split_dirs)

            # 실제 처리 성공률 계산
            total_videos = sum(len(videos) for videos in video_splits)
            processed_videos = 0

            # Stage1 poses 파일 수로 실제 처리된 비디오 수 계산
            stage1_dir = Path(self.output_dir) / "stage1_poses"
            if stage1_dir.exists():
                poses_files = list(stage1_dir.rglob("*_poses.pkl"))
                processed_videos = len(poses_files)

            actual_success_rate = (processed_videos / total_videos * 100) if total_videos > 0 else 0

            total_time = time.time() - start_time
            logger.info(f"Multi-process annotation completed in {total_time:.2f}s")
            logger.info(f"Process success rate: {process_success_count}/{len(processes)}")
            logger.info(f"Video processing success rate: {processed_videos}/{total_videos} ({actual_success_rate:.1f}%)")

            # Stage3 결과 확인
            if stage_results.get('stage3', False):
                stage3_files = [
                    Path(self.output_dir) / "stage3_dataset" / "train.pkl",
                    Path(self.output_dir) / "stage3_dataset" / "val.pkl",
                    Path(self.output_dir) / "stage3_dataset" / "test.pkl"
                ]
                stage3_success = all(f.exists() for f in stage3_files)
                logger.info(f"Stage3 dataset generation: {'✅ Success' if stage3_success else '❌ Failed'}")

            # Stage3 성공 여부 확인
            stage3_success = stage_results.get('stage3', False)

            if stage3_success:
                logger.info("✅ Stage3 dataset generation completed successfully!")
            else:
                logger.warning("⚠️ Stage3 dataset generation failed")

            # stage1 멀티프로세스만 실행하므로 stage1 성공률로 판단
            overall_success = actual_success_rate >= 80.0

            if overall_success:
                logger.info("🎉 Multi-process stage1: OVERALL SUCCESS!")
            else:
                logger.warning("⚠️ Multi-process stage1: Partial success or failure")

            return overall_success
            
        except Exception as e:
            logger.error(f"Multi-process annotation failed: {e}")
            return False


# 편의 함수
def run_multi_process_annotation(
    input_dir: str,
    output_dir: str,
    config_path: str,
    num_processes: int = 4,
    gpu_assignments: List[int] = None
) -> bool:
    """멀티 프로세스 어노테이션 실행"""
    
    manager = MultiProcessAnnotationManager(
        input_dir=input_dir,
        output_dir=output_dir,
        config_path=config_path,
        num_processes=num_processes,
        gpu_assignments=gpu_assignments
    )
    
    return manager.run_full_pipeline()


class MultiProcessInferenceAnalysisManager:
    """멀티 프로세스 inference.analysis 관리자"""
    
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        config_path: str,
        num_processes: int = 4,
        gpu_assignments: List[int] = None
    ):
        self.input_dir = input_dir
        self.config_path = config_path
        self.num_processes = num_processes
        self.gpu_assignments = gpu_assignments or [i % 2 for i in range(num_processes)]
        
        # 출력 디렉토리를 입력 폴더명으로 설정
        input_folder_name = Path(input_dir).name
        self.output_dir = str(Path(output_dir) / input_folder_name)
        
        # 컴포넌트 초기화
        self.splitter = VideoDataSplitter(input_dir, num_processes)
        self.config_generator = InferenceAnalysisConfigGenerator(config_path)
        self.runner = MultiProcessRunner()
        self.merger = InferenceResultMerger(self.output_dir)
    
    def run_full_pipeline(self) -> bool:
        """전체 파이프라인 실행"""
        start_time = time.time()
        
        try:
            logger.info(f"Starting multi-process inference analysis with {self.num_processes} processes")
            logger.info(f"Input: {self.input_dir}")
            logger.info(f"Output: {self.output_dir}")
            logger.info(f"GPU assignments: {self.gpu_assignments}")
            
            # 1. 비디오 분할
            logger.info("=== Step 1: Splitting videos ===")
            video_splits = self.splitter.split_videos()
            if not video_splits:
                logger.error("No videos to process")
                return False
            
            # 2. 임시 디렉토리 생성 (입력 디렉토리 기반으로)
            input_parent = Path(self.input_dir).parent
            input_folder_name = Path(self.input_dir).name
            temp_base = input_parent / f"{input_folder_name}_temp_splits"
            split_dirs = self.splitter.create_split_directories(str(temp_base))
            
            # 3. 분할별 설정 파일 생성
            logger.info("=== Step 2: Creating split configurations ===")
            config_paths = self.config_generator.create_split_configs(video_splits, split_dirs)
            
            # 4. 멀티 프로세스 실행
            logger.info("=== Step 3: Running split processes ===")
            processes = self.runner.run_split_processes(config_paths, self.gpu_assignments)
            
            # 5. 프로세스 모니터링
            logger.info("=== Step 4: Monitoring processes ===")
            results = self.runner.monitor_processes(processes)
            
            # 6. 결과 통합
            logger.info("=== Step 5: Merging results ===")
            success_count = sum(1 for code in results.values() if code == 0)
            logger.info(f"Completed processes: {success_count}/{len(processes)}")
            
            # inference.analysis 결과 통합
            self.merger.merge_inference_results(split_dirs)
            
            # 7. 정리
            logger.info("=== Step 6: Cleanup ===")
            self.merger.cleanup_split_dirs(split_dirs)
            
            total_time = time.time() - start_time
            logger.info(f"Multi-process inference analysis completed in {total_time:.2f}s")
            logger.info(f"Success rate: {success_count}/{len(processes)}")
            
            return success_count == len(processes)
            
        except Exception as e:
            logger.error(f"Multi-process inference analysis failed: {e}")
            return False


class InferenceAnalysisConfigGenerator:
    """inference.analysis용 설정 파일 생성기"""
    
    def __init__(self, base_config_path: str):
        self.base_config_path = base_config_path
        with open(base_config_path, 'r', encoding='utf-8') as f:
            self.base_config = yaml.safe_load(f)
    
    def create_split_configs(
        self, 
        video_splits: List[List[Path]], 
        split_dirs: List[str]
    ) -> List[str]:
        """분할별 설정 파일 생성"""
        config_paths = []
        
        for i, (videos, split_dir) in enumerate(zip(video_splits, split_dirs)):
            # 임시 비디오 디렉토리 생성
            temp_video_dir = Path(split_dir) / "videos"
            temp_video_dir.mkdir(exist_ok=True)
            
            # 비디오 파일 심볼릭 링크 생성 (원본 폴더 구조 보존)
            for video in videos:
                # 원본 폴더명 보존 (fight, normal 등)
                original_parent = video.parent.name
                target_subdir = temp_video_dir / original_parent
                target_subdir.mkdir(exist_ok=True)
                
                link_path = target_subdir / video.name
                if not link_path.exists():
                    try:
                        os.symlink(str(video.absolute()), str(link_path))
                    except:
                        # 심볼릭 링크 실패시 복사
                        shutil.copy2(str(video), str(link_path))
            
            # 분할별 설정 생성
            split_config = self.base_config.copy()
            
            # inference.analysis 모드로 설정
            split_config['mode'] = 'inference.analysis'
            split_config['inference']['analysis']['input'] = str(temp_video_dir)
            split_config['inference']['analysis']['output_dir'] = split_dir
            
            # 멀티 프로세스 비활성화 (subprocess에서는 단일 처리만)
            split_config['multi_process']['enabled'] = False
            
            # 성능평가는 마지막에만 (첫 번째 프로세스에서만 실행)
            split_config['inference']['analysis']['enable_evaluation'] = (i == 0)
            
            # 설정 파일 저장
            config_path = Path(split_dir) / f"config_split_{i}.yaml"
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(split_config, f, allow_unicode=True)
            
            config_paths.append(str(config_path))
            logger.info(f"Created config for split {i}: {len(videos)} videos -> {config_path}")
        
        return config_paths


class InferenceResultMerger:
    """inference.analysis 결과 통합기"""
    
    def __init__(self, final_output_dir: str):
        self.final_output_dir = Path(final_output_dir)
        self.final_output_dir.mkdir(parents=True, exist_ok=True)
    
    def merge_inference_results(self, split_dirs: List[str]) -> bool:
        """inference.analysis 결과 통합 (JSON + PKL 파일)"""
        try:
            merged_count = 0
            pkl_merged_count = 0
            
            for split_dir in split_dirs:
                split_path = Path(split_dir)
                
                # JSON 결과 파일들 찾기
                json_files = list(split_path.rglob("**/*_results.json"))
                
                for json_file in json_files:
                    try:
                        # 상대 경로 계산하되 videos/ 부분을 제거
                        rel_path = json_file.relative_to(split_path)
                        
                        # videos/ 경로를 제거하고 재구성
                        path_parts = list(rel_path.parts)
                        if path_parts and path_parts[0] == 'videos':
                            # videos/F_20_1_1_0_0/F_20_1_1_0_0_results.json -> F_20_1_1_0_0/F_20_1_1_0_0_results.json
                            new_rel_path = Path(*path_parts[1:])
                        else:
                            new_rel_path = rel_path
                        
                        final_path = self.final_output_dir / new_rel_path
                        
                        # 디렉토리 생성
                        final_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # 파일 이동 (또는 복사)
                        shutil.move(str(json_file), str(final_path))
                        merged_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Failed to merge {json_file}: {e}")
                
                # PKL 파일들 찾기 및 통합 (analysis 모드이므로 필수)
                pkl_files = list(split_path.rglob("**/*.pkl"))
                
                for pkl_file in pkl_files:
                    try:
                        # 상대 경로 계산하되 videos/ 부분을 제거
                        rel_path = pkl_file.relative_to(split_path)
                        
                        # videos/ 경로를 제거하고 재구성
                        path_parts = list(rel_path.parts)
                        if path_parts and path_parts[0] == 'videos':
                            new_rel_path = Path(*path_parts[1:])
                        else:
                            new_rel_path = rel_path
                        
                        final_path = self.final_output_dir / new_rel_path
                        
                        # 디렉토리 생성
                        final_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # 파일 이동 (또는 복사)
                        shutil.move(str(pkl_file), str(final_path))
                        pkl_merged_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Failed to merge {pkl_file}: {e}")
                
                # evaluation 결과가 있으면 통합
                eval_dir = split_path / "evaluation"
                if eval_dir.exists():
                    final_eval_dir = self.final_output_dir / "evaluation"
                    if final_eval_dir.exists():
                        shutil.rmtree(str(final_eval_dir))
                    shutil.move(str(eval_dir), str(final_eval_dir))
                    logger.info("Merged evaluation results")
            
            logger.info(f"Merged {merged_count} inference result files")
            logger.info(f"Merged {pkl_merged_count} PKL files")
            return merged_count > 0
            
        except Exception as e:
            logger.error(f"Error merging inference results: {e}")
            return False
    
    def cleanup_split_dirs(self, split_dirs: List[str]):
        """분할 디렉토리 완전 정리"""
        # temp_splits 폴더 전체 삭제
        if split_dirs:
            temp_base = Path(split_dirs[0]).parent
            try:
                if temp_base.name.endswith("_temp_splits"):
                    shutil.rmtree(str(temp_base))
                    logger.info(f"Cleaned up temp_splits directory: {temp_base}")
                    return
            except Exception as e:
                logger.warning(f"Failed to cleanup temp_splits: {e}")
        
        # 개별 분할 디렉토리 삭제
        for split_dir in split_dirs:
            try:
                shutil.rmtree(split_dir)
                logger.info(f"Cleaned up split directory: {split_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {split_dir}: {e}")


def run_multi_process_inference_analysis(
    input_dir: str,
    output_dir: str,
    config_path: str,
    num_processes: int = 4,
    gpu_assignments: List[int] = None
) -> bool:
    """멀티 프로세스 inference.analysis 실행"""
    
    manager = MultiProcessInferenceAnalysisManager(
        input_dir=input_dir,
        output_dir=output_dir,
        config_path=config_path,
        num_processes=num_processes,
        gpu_assignments=gpu_assignments
    )
    
    return manager.run_full_pipeline()


if __name__ == "__main__":
    # 테스트 실행
    success = run_multi_process_annotation(
        input_dir="/aivanas/raw/surveillance/action/violence/action_recognition/data/RWF-2000",
        output_dir="/workspace/recognizer/output/RWF-2000",
        config_path="/workspace/recognizer/configs/config.yaml",
        num_processes=4,
        gpu_assignments=[0, 1, 0, 1]  # GPU 0, 1 순환 할당
    )
    
    if success:
        print("Multi-process annotation completed successfully!")
    else:
        print("Multi-process annotation failed!")