#!/usr/bin/env python3
"""
ONNX GPU 최적화 자동 벤치마킹 도구

GPU별로 최적의 ONNXRuntime 설정을 자동으로 찾아주는 도구입니다.
다양한 하드웨어 환경에서 최고 성능을 달성할 수 있도록 설정을 최적화합니다.
"""

import os
import sys
import time
import json
import yaml
import argparse
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

# ONNXRuntime 관련 import
try:
    import onnxruntime as ort
    import torch
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


@dataclass
class OptimizationConfig:
    """최적화 설정 구조"""
    # cuDNN 설정
    cudnn_conv_algo_search: str  # 'DEFAULT', 'HEURISTIC', 'EXHAUSTIVE'
    do_copy_in_default_stream: bool
    cudnn_conv_use_max_workspace: bool
    tunable_op_enable: bool
    tunable_op_tuning_enable: bool
    
    # GPU 메모리 설정
    gpu_mem_limit_gb: int  # GB 단위
    arena_extend_strategy: str  # 'kNextPowerOfTwo', 'kSameAsRequested'
    
    # 세션 설정
    execution_mode: str  # 'ORT_SEQUENTIAL', 'ORT_PARALLEL'
    graph_optimization_level: str  # 'ORT_ENABLE_ALL', 'ORT_ENABLE_BASIC'
    enable_cpu_mem_arena: bool
    enable_mem_pattern: bool
    
    def to_provider_options(self) -> Dict[str, Any]:
        """ONNXRuntime Provider 옵션으로 변환"""
        return {
            'device_id': 0,
            'gpu_mem_limit': self.gpu_mem_limit_gb * 1024 * 1024 * 1024,
            'arena_extend_strategy': self.arena_extend_strategy,
            'cudnn_conv_algo_search': self.cudnn_conv_algo_search,
            'do_copy_in_default_stream': self.do_copy_in_default_stream,
            'cudnn_conv_use_max_workspace': self.cudnn_conv_use_max_workspace,
            'tunable_op_enable': self.tunable_op_enable,
            'tunable_op_tuning_enable': self.tunable_op_tuning_enable,
        }
    
    def to_session_options(self) -> ort.SessionOptions:
        """ONNXRuntime 세션 옵션으로 변환"""
        session_options = ort.SessionOptions()
        session_options.execution_mode = getattr(ort.ExecutionMode, self.execution_mode)
        session_options.graph_optimization_level = getattr(ort.GraphOptimizationLevel, self.graph_optimization_level)
        session_options.enable_cpu_mem_arena = self.enable_cpu_mem_arena
        session_options.enable_mem_pattern = self.enable_mem_pattern
        session_options.log_severity_level = 3  # 경고 줄이기
        return session_options


@dataclass
class BenchmarkResult:
    """벤치마크 결과 구조"""
    config: OptimizationConfig
    avg_time_ms: float
    std_time_ms: float
    fps: float
    success: bool
    error_message: Optional[str] = None
    gpu_provider_active: bool = False
    
    def score(self) -> float:
        """성능 점수 계산 (FPS 기반, 실패시 0)"""
        return self.fps if self.success and self.gpu_provider_active else 0.0


class ONNXOptimizerBenchmark:
    """ONNX 최적화 벤치마크 클래스"""
    
    def __init__(self, model_path: str, warmup_runs: int = 20, test_runs: int = 50):
        self.model_path = model_path
        self.warmup_runs = warmup_runs
        self.test_runs = test_runs
        self.logger = self._setup_logger()
        
        # 환경 정보 수집
        self.gpu_info = self._get_gpu_info()
        self.system_info = self._get_system_info()
        
    def _setup_logger(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger('ONNXOptimizer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def _get_gpu_info(self) -> Dict[str, Any]:
        """GPU 정보 수집"""
        if not torch.cuda.is_available():
            return {'available': False}
            
        gpu_props = torch.cuda.get_device_properties(0)
        return {
            'available': True,
            'name': gpu_props.name,
            'total_memory_gb': gpu_props.total_memory / 1024**3,
            'multi_processor_count': gpu_props.multi_processor_count,
            'compute_capability': torch.cuda.get_device_capability(0),
            'cuda_version': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version(),
        }
        
    def _get_system_info(self) -> Dict[str, Any]:
        """시스템 정보 수집"""
        return {
            'onnxruntime_version': ort.__version__ if HAS_ONNX else 'N/A',
            'python_version': sys.version.split()[0],
            'platform': sys.platform,
        }
    
    def generate_config_combinations(self) -> List[OptimizationConfig]:
        """최적화 설정 조합 생성"""
        combinations = []
        
        # GPU 메모리 기반 설정
        available_memory = self.gpu_info.get('total_memory_gb', 24)
        memory_limits = [
            min(4, int(available_memory * 0.2)),
            min(8, int(available_memory * 0.4)), 
            min(16, int(available_memory * 0.7)),
            min(24, int(available_memory * 0.9)),
        ]
        
        # 주요 설정 조합
        base_configs = [
            # 빠른 휴리스틱 기반
            {
                'cudnn_conv_algo_search': 'HEURISTIC',
                'do_copy_in_default_stream': False,
                'cudnn_conv_use_max_workspace': True,
                'tunable_op_enable': True,
                'tunable_op_tuning_enable': True,
                'execution_mode': 'ORT_SEQUENTIAL',
                'graph_optimization_level': 'ORT_ENABLE_ALL',
                'enable_cpu_mem_arena': True,
                'enable_mem_pattern': True,
                'arena_extend_strategy': 'kNextPowerOfTwo',
            },
            # 전수탐색 기반
            {
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': False,
                'cudnn_conv_use_max_workspace': True,
                'tunable_op_enable': False,
                'tunable_op_tuning_enable': False,
                'execution_mode': 'ORT_SEQUENTIAL',
                'graph_optimization_level': 'ORT_ENABLE_ALL',
                'enable_cpu_mem_arena': True,
                'enable_mem_pattern': True,
                'arena_extend_strategy': 'kNextPowerOfTwo',
            },
            # 보수적 설정
            {
                'cudnn_conv_algo_search': 'DEFAULT',
                'do_copy_in_default_stream': True,
                'cudnn_conv_use_max_workspace': False,
                'tunable_op_enable': False,
                'tunable_op_tuning_enable': False,
                'execution_mode': 'ORT_SEQUENTIAL',
                'graph_optimization_level': 'ORT_ENABLE_BASIC',
                'enable_cpu_mem_arena': True,
                'enable_mem_pattern': True,
                'arena_extend_strategy': 'kSameAsRequested',
            },
        ]
        
        # 메모리 크기별로 각 설정 조합
        for base_config in base_configs:
            for mem_limit in memory_limits:
                config = OptimizationConfig(
                    gpu_mem_limit_gb=mem_limit,
                    **base_config
                )
                combinations.append(config)
                
        # 추가 실험적 조합
        experimental_configs = [
            # 병렬 실행 모드
            OptimizationConfig(
                cudnn_conv_algo_search='HEURISTIC',
                do_copy_in_default_stream=False,
                cudnn_conv_use_max_workspace=True,
                tunable_op_enable=True,
                tunable_op_tuning_enable=True,
                gpu_mem_limit_gb=memory_limits[-1],
                arena_extend_strategy='kNextPowerOfTwo',
                execution_mode='ORT_PARALLEL',  # 병렬 모드
                graph_optimization_level='ORT_ENABLE_ALL',
                enable_cpu_mem_arena=False,  # 병렬에서는 False
                enable_mem_pattern=False,
            ),
            # 메모리 최적화 모드
            OptimizationConfig(
                cudnn_conv_algo_search='HEURISTIC',
                do_copy_in_default_stream=True,
                cudnn_conv_use_max_workspace=False,
                tunable_op_enable=False,
                tunable_op_tuning_enable=False,
                gpu_mem_limit_gb=memory_limits[0],  # 최소 메모리
                arena_extend_strategy='kSameAsRequested',
                execution_mode='ORT_SEQUENTIAL',
                graph_optimization_level='ORT_ENABLE_ALL',
                enable_cpu_mem_arena=True,
                enable_mem_pattern=True,
            ),
        ]
        
        combinations.extend(experimental_configs)
        return combinations
    
    def benchmark_single_config(self, config: OptimizationConfig) -> BenchmarkResult:
        """단일 설정에 대한 벤치마크 수행"""
        try:
            # 세션 생성
            session_options = config.to_session_options()
            provider_options = config.to_provider_options()
            
            providers = [
                ('CUDAExecutionProvider', provider_options),
                'CPUExecutionProvider'
            ]
            
            session = ort.InferenceSession(
                self.model_path,
                providers=providers,
                sess_options=session_options
            )
            
            # GPU 제공자 활성화 확인
            gpu_active = 'CUDAExecutionProvider' in session.get_providers()
            
            if not gpu_active:
                return BenchmarkResult(
                    config=config,
                    avg_time_ms=0,
                    std_time_ms=0,
                    fps=0,
                    success=False,
                    error_message="CUDA Provider 활성화 실패",
                    gpu_provider_active=False
                )
            
            # 테스트 입력 준비
            input_tensor = np.random.rand(1, 3, 640, 640).astype(np.float32)
            input_name = session.get_inputs()[0].name
            
            # 워밍업 (충분한 시간으로 알고리즘 안정화)
            self.logger.info(f"워밍업 중... (알고리즘: {config.cudnn_conv_algo_search})")
            for _ in range(self.warmup_runs):
                session.run(None, {input_name: input_tensor})
                
            # 성능 측정
            times = []
            for _ in range(self.test_runs):
                start_time = time.time()
                session.run(None, {input_name: input_tensor})
                times.append(time.time() - start_time)
            
            # 통계 계산
            avg_time = np.mean(times)
            std_time = np.std(times)
            avg_time_ms = avg_time * 1000
            std_time_ms = std_time * 1000
            fps = 1000 / avg_time_ms
            
            # 세션 정리
            session = None
            
            return BenchmarkResult(
                config=config,
                avg_time_ms=avg_time_ms,
                std_time_ms=std_time_ms,
                fps=fps,
                success=True,
                gpu_provider_active=gpu_active
            )
            
        except Exception as e:
            self.logger.error(f"벤치마크 실패: {str(e)}")
            return BenchmarkResult(
                config=config,
                avg_time_ms=0,
                std_time_ms=0,
                fps=0,
                success=False,
                error_message=str(e),
                gpu_provider_active=False
            )
    
    def run_full_benchmark(self) -> List[BenchmarkResult]:
        """전체 벤치마크 실행"""
        if not HAS_ONNX:
            raise RuntimeError("ONNXRuntime이 설치되지 않았습니다.")
            
        if not self.gpu_info['available']:
            raise RuntimeError("CUDA GPU를 사용할 수 없습니다.")
            
        self.logger.info("ONNX GPU 최적화 벤치마크 시작")
        self.logger.info(f"GPU: {self.gpu_info['name']}")
        self.logger.info(f"모델: {self.model_path}")
        
        # 설정 조합 생성
        configs = self.generate_config_combinations()
        self.logger.info(f"총 {len(configs)}개 설정 조합 테스트 예정")
        
        results = []
        
        for i, config in enumerate(configs, 1):
            self.logger.info(f"[{i}/{len(configs)}] 테스트 중: {config.cudnn_conv_algo_search} + {config.gpu_mem_limit_gb}GB")
            
            try:
                result = self.benchmark_single_config(config)
                results.append(result)
                
                if result.success and result.gpu_provider_active:
                    self.logger.info(f"  결과: {result.avg_time_ms:.2f}ms ({result.fps:.1f} FPS)")
                else:
                    self.logger.warning(f"  실패: {result.error_message}")
                    
            except Exception as e:
                self.logger.error(f"  예외 발생: {str(e)}")
                result = BenchmarkResult(
                    config=config,
                    avg_time_ms=0,
                    std_time_ms=0,
                    fps=0,
                    success=False,
                    error_message=str(e),
                    gpu_provider_active=False
                )
                results.append(result)
        
        return results
    
    def find_optimal_config(self, results: List[BenchmarkResult]) -> BenchmarkResult:
        """최적 설정 찾기"""
        successful_results = [r for r in results if r.success and r.gpu_provider_active]
        
        if not successful_results:
            raise RuntimeError("성공한 벤치마크가 없습니다.")
            
        # FPS 기준으로 정렬
        successful_results.sort(key=lambda x: x.fps, reverse=True)
        
        return successful_results[0]
    
    def generate_report(self, results: List[BenchmarkResult], optimal: BenchmarkResult) -> str:
        """벤치마크 보고서 생성"""
        successful_results = [r for r in results if r.success and r.gpu_provider_active]
        
        report = []
        report.append("# ONNX GPU 최적화 벤치마크 보고서")
        report.append("")
        
        # 시스템 정보
        report.append("## 시스템 정보")
        report.append("")
        report.append(f"- **GPU**: {self.gpu_info['name']}")
        report.append(f"- **GPU 메모리**: {self.gpu_info['total_memory_gb']:.1f} GB")
        report.append(f"- **CUDA**: {self.gpu_info['cuda_version']}")
        report.append(f"- **cuDNN**: {self.gpu_info['cudnn_version']}")
        report.append(f"- **ONNXRuntime**: {self.system_info['onnxruntime_version']}")
        report.append("")
        
        # 최적 설정
        report.append("## 🏆 최적 설정")
        report.append("")
        report.append(f"**성능**: {optimal.avg_time_ms:.2f}ms ({optimal.fps:.1f} FPS)")
        report.append("")
        report.append("```yaml")
        report.append("# config.yaml에 적용할 설정")
        report.append("models:")
        report.append("  pose_estimation:")
        report.append("    onnx:")
        optimal_dict = asdict(optimal.config)
        for key, value in optimal_dict.items():
            if isinstance(value, str):
                report.append(f"      {key}: \"{value}\"")
            else:
                report.append(f"      {key}: {value}")
        report.append("```")
        report.append("")
        
        # 성능 비교표
        report.append("## 📊 성능 비교")
        report.append("")
        report.append("| 순위 | 알고리즘 | FPS | 시간(ms) | GPU메모리(GB) | 스트림 | 작업공간 |")
        report.append("|------|----------|-----|----------|---------------|--------|----------|")
        
        for i, result in enumerate(successful_results[:10], 1):
            config = result.config
            report.append(f"| {i} | {config.cudnn_conv_algo_search} | {result.fps:.1f} | {result.avg_time_ms:.2f} | {config.gpu_mem_limit_gb} | {config.do_copy_in_default_stream} | {config.cudnn_conv_use_max_workspace} |")
        
        report.append("")
        
        # 설정별 분석
        report.append("## 🔍 설정별 분석")
        report.append("")
        
        # 알고리즘별 성능
        algo_stats = {}
        for result in successful_results:
            algo = result.config.cudnn_conv_algo_search
            if algo not in algo_stats:
                algo_stats[algo] = []
            algo_stats[algo].append(result.fps)
        
        for algo, fps_list in algo_stats.items():
            avg_fps = np.mean(fps_list)
            max_fps = np.max(fps_list)
            report.append(f"- **{algo}**: 평균 {avg_fps:.1f} FPS, 최고 {max_fps:.1f} FPS")
        
        report.append("")
        
        # 권장사항
        report.append("## 💡 권장사항")
        report.append("")
        
        if optimal.config.cudnn_conv_algo_search == 'HEURISTIC':
            report.append("✅ **HEURISTIC 알고리즘 권장**")
            report.append("- 빠른 초기화와 우수한 성능의 균형")
            report.append("- 실시간 추론에 최적화")
            report.append("- 프로덕션 환경에 권장")
        elif optimal.config.cudnn_conv_algo_search == 'EXHAUSTIVE':
            report.append("✅ **EXHAUSTIVE 알고리즘 권장**") 
            report.append("- 최적 성능을 위한 전수탐색")
            report.append("- 초기화 시간이 길지만 안정화 후 최고 성능")
            report.append("- 배치 처리나 장시간 실행에 권장")
        else:
            report.append("✅ **DEFAULT 알고리즘 권장**")
            report.append("- 기본 설정으로 안정적인 성능")
            
        report.append("")
        report.append(f"**메모리 설정**: {optimal.config.gpu_mem_limit_gb}GB 할당")
        report.append(f"**스트림 모드**: {'기본 스트림 사용' if optimal.config.do_copy_in_default_stream else '별도 스트림 사용'}")
        
        return "\n".join(report)
    
    def update_config_file(self, config_path: str, optimal_config: OptimizationConfig):
        """config.yaml 파일 자동 업데이트"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # ONNX 설정 업데이트
            onnx_config = config['models']['pose_estimation']['onnx']
            
            # 최적화 설정 적용
            onnx_config.update({
                'cudnn_conv_algo_search': optimal_config.cudnn_conv_algo_search,
                'do_copy_in_default_stream': optimal_config.do_copy_in_default_stream,
                'cudnn_conv_use_max_workspace': optimal_config.cudnn_conv_use_max_workspace,
                'tunable_op_enable': optimal_config.tunable_op_enable,
                'tunable_op_tuning_enable': optimal_config.tunable_op_tuning_enable,
                'gpu_mem_limit_gb': optimal_config.gpu_mem_limit_gb,
                'arena_extend_strategy': optimal_config.arena_extend_strategy,
                'execution_mode': optimal_config.execution_mode,
                'graph_optimization_level': optimal_config.graph_optimization_level,
                'enable_cpu_mem_arena': optimal_config.enable_cpu_mem_arena,
                'enable_mem_pattern': optimal_config.enable_mem_pattern,
            })
            
            # 성능 정보 추가
            onnx_config['benchmark_info'] = {
                'gpu_name': self.gpu_info['name'],
                'optimized_fps': round(optimal_config.to_provider_options().get('fps', 0), 1),  # 이 부분은 수정 필요
                'benchmark_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'onnxruntime_version': self.system_info['onnxruntime_version'],
            }
            
            # 파일 저장
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
                
            self.logger.info(f"설정 파일 업데이트 완료: {config_path}")
            
        except Exception as e:
            self.logger.error(f"설정 파일 업데이트 실패: {str(e)}")
            raise


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='ONNX GPU 최적화 벤치마킹 도구')
    parser.add_argument('--model', required=True, help='ONNX 모델 파일 경로')
    parser.add_argument('--config', help='업데이트할 config.yaml 경로')
    parser.add_argument('--output', help='보고서 출력 파일 경로 (.md)')
    parser.add_argument('--warmup', type=int, default=20, help='워밍업 횟수')
    parser.add_argument('--runs', type=int, default=50, help='측정 횟수')
    parser.add_argument('--quick', action='store_true', help='빠른 테스트 (적은 조합)')
    
    args = parser.parse_args()
    
    # 입력 검증
    if not os.path.exists(args.model):
        print(f"모델 파일을 찾을 수 없습니다: {args.model}")
        sys.exit(1)
    
    # 벤치마크 실행
    benchmark = ONNXOptimizerBenchmark(
        model_path=args.model,
        warmup_runs=args.warmup,
        test_runs=args.runs
    )
    
    print("🚀 ONNX GPU 최적화 벤치마크 시작")
    print(f"GPU: {benchmark.gpu_info['name']}")
    print(f"모델: {args.model}")
    print()
    
    # 전체 벤치마크 실행
    results = benchmark.run_full_benchmark()
    
    # 최적 설정 찾기
    try:
        optimal = benchmark.find_optimal_config(results)
        
        print("🏆 최적 설정 발견!")
        print(f"성능: {optimal.avg_time_ms:.2f}ms ({optimal.fps:.1f} FPS)")
        print(f"알고리즘: {optimal.config.cudnn_conv_algo_search}")
        print(f"GPU 메모리: {optimal.config.gpu_mem_limit_gb}GB")
        
        # 보고서 생성
        report = benchmark.generate_report(results, optimal)
        
        # 보고서 저장
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"보고서 저장: {args.output}")
        else:
            print("\n" + "="*60)
            print(report)
        
        # 설정 파일 업데이트
        if args.config:
            if os.path.exists(args.config):
                benchmark.update_config_file(args.config, optimal.config)
                print(f"설정 파일 업데이트: {args.config}")
            else:
                print(f"설정 파일을 찾을 수 없습니다: {args.config}")
        
    except RuntimeError as e:
        print(f"오류: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()