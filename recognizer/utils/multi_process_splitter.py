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
            
            # 멀티 프로세스 비활성화 (subprocess에서는 단일 처리만)
            if 'multi_process' in split_config['annotation']:
                split_config['annotation']['multi_process']['enabled'] = False
            
            # 병렬 처리 비활성화 (순차 처리)
            if 'stage1' in split_config['annotation']:
                split_config['annotation']['stage1']['enable_parallel'] = False
            if 'stage2' in split_config['annotation']:
                split_config['annotation']['stage2']['enable_parallel'] = False
            
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
        """프로세스 병렬 모니터링"""
        import threading
        import queue
        
        results = {}
        result_queue = queue.Queue()
        
        def monitor_single_process(i, process):
            """단일 프로세스 모니터링 스레드"""
            try:
                logger.info(f"Monitoring split {i} (PID: {process.pid})")
                
                # 실시간 출력 모니터링
                for line in process.stdout:
                    logger.info(f"Split {i}: {line.strip()}")
                
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
        
        # 각 프로세스별로 모니터링 스레드 시작
        monitor_threads = []
        for i, process in enumerate(processes):
            thread = threading.Thread(target=monitor_single_process, args=(i, process))
            thread.daemon = True
            thread.start()
            monitor_threads.append(thread)
        
        # 모든 결과 수집
        for _ in range(len(processes)):
            i, return_code = result_queue.get()
            results[i] = return_code
        
        # 모든 스레드 완료 대기
        for thread in monitor_threads:
            thread.join()
        
        return results


class ResultMerger:
    """결과 통합기"""
    
    def __init__(self, final_output_dir: str):
        self.final_output_dir = Path(final_output_dir)
        self.final_output_dir.mkdir(parents=True, exist_ok=True)
    
    def merge_stage_results(self, split_dirs: List[str], stage: str) -> bool:
        """스테이지별 결과 통합"""
        try:
            # 최종 출력 디렉토리 구조 생성
            stage_patterns = {
                'stage1': '**/stage1_poses/**/*.pkl',
                'stage2': '**/stage2_tracking/**/*.pkl', 
                'stage3': '**/stage3_dataset/**/*.pkl'
            }
            
            pattern = stage_patterns.get(stage, f'**/{stage}**/*.pkl')
            merged_count = 0
            
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
                        
                        # 디렉토리 생성
                        final_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # 파일 이동 (또는 복사)
                        shutil.move(str(result_file), str(final_path))
                        merged_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Failed to merge {result_file}: {e}")
            
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
                if temp_base.name == "temp_splits":
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
            
            # 2. 임시 디렉토리 생성 (상위 디렉토리에)
            temp_base = Path(self.output_dir).parent / "temp_splits"
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
            
            # 각 스테이지별 결과 통합
            for stage in ['stage1', 'stage2', 'stage3']:
                self.merger.merge_stage_results(split_dirs, stage)
            
            # 7. 정리
            logger.info("=== Step 6: Cleanup ===")
            self.merger.cleanup_split_dirs(split_dirs)
            
            total_time = time.time() - start_time
            logger.info(f"Multi-process annotation completed in {total_time:.2f}s")
            logger.info(f"Success rate: {success_count}/{len(processes)}")
            
            return success_count == len(processes)
            
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
            
            # 2. 임시 디렉토리 생성 (상위 디렉토리에)
            temp_base = Path(self.output_dir).parent / "temp_splits"
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
                if temp_base.name == "temp_splits":
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