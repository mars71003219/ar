# ONNX 최적화 인자 완전 참조 가이드

14가지 조합에 사용된 각 인자의 정의, 동작 원리, 성능 영향을 상세히 설명합니다.

## 📋 인자 분류

### 🧠 핵심 알고리즘 선택
- `cudnn_conv_algo_search`: cuDNN 컨볼루션 알고리즘 선택 방식
- `execution_mode`: ONNXRuntime 실행 모드

### 💾 메모리 관리  
- `gpu_mem_limit_gb`: GPU 메모리 할당 한계
- `arena_extend_strategy`: 메모리 아레나 확장 전략
- `enable_cpu_mem_arena`: CPU 메모리 아레나 활성화
- `enable_mem_pattern`: 메모리 패턴 최적화

### ⚡ 성능 최적화
- `do_copy_in_default_stream`: CUDA 스트림 모드
- `cudnn_conv_use_max_workspace`: cuDNN 작업공간 크기
- `tunable_op_enable`: 동적 연산 최적화
- `graph_optimization_level`: 그래프 최적화 수준

---

## 🧠 알고리즘 선택 인자

### `cudnn_conv_algo_search`

cuDNN이 컨볼루션 연산에 사용할 알고리즘을 선택하는 방식을 결정합니다.

#### 옵션별 상세 동작

**`"HEURISTIC"` (휴리스틱)**
```cpp
// cuDNN 내부 의사코드
Algorithm selectConvolutionAlgorithm(TensorDesc input, FilterDesc filter) {
    // 1. 입력 특성 분석
    int input_size = input.height * input.width;
    int channels = input.channels;
    int filter_size = filter.height * filter.width;
    
    // 2. GPU 아키텍처 고려
    if (gpu_arch == AMPERE && input_size > 300*300) {
        if (filter_size <= 3) {
            return ALGO_WINOGRAD_NONFUSED;  // 큰 입력 + 작은 필터
        } else {
            return ALGO_IMPLICIT_GEMM;      // 큰 입력 + 큰 필터  
        }
    } else if (gpu_arch == TURING) {
        return ALGO_DIRECT_CONVOLUTION;     // 안정적 선택
    }
    
    // 3. 메모리 대역폭 고려
    if (memory_bandwidth_limited) {
        return memory_efficient_algorithm();
    } else {
        return compute_intensive_algorithm();
    }
}
```
- **장점**: 빠른 선택 (1-2ms), 실용적 성능, 메모리 효율적
- **단점**: 이론적 최적이 아닐 수 있음
- **적합한 환경**: 실시간 추론, 프로덕션 환경

**`"EXHAUSTIVE"` (전수탐색)**
```cpp
// cuDNN 내부 의사코드  
Algorithm selectConvolutionAlgorithm(TensorDesc input, FilterDesc filter) {
    vector<Algorithm> available_algos = {
        ALGO_IMPLICIT_GEMM,
        ALGO_IMPLICIT_PRECOMP_GEMM, 
        ALGO_GEMM,
        ALGO_DIRECT,
        ALGO_FFT,
        ALGO_FFT_TILING,
        ALGO_WINOGRAD,
        ALGO_WINOGRAD_NONFUSED
    };
    
    Algorithm best_algo;
    float best_time = INFINITY;
    
    // 모든 알고리즘 실제 실행하여 측정
    for (auto algo : available_algos) {
        try {
            float exec_time = benchmark_algorithm(algo, input, filter, actual_data);
            if (exec_time < best_time) {
                best_time = exec_time;
                best_algo = algo;
            }
        } catch (OutOfMemoryError) {
            continue;  // 메모리 부족시 스킵
        }
    }
    
    return best_algo;  // 실제 측정으로 찾은 최적 알고리즘
}
```
- **장점**: 이론적 최적 알고리즘 선택
- **단점**: 초기화 오래 걸림 (2-5초), 메모리 많이 사용
- **적합한 환경**: 배치 처리, 장시간 실행

**`"DEFAULT"` (기본)**
```cpp
// cuDNN 내부 의사코드
Algorithm selectConvolutionAlgorithm(TensorDesc input, FilterDesc filter) {
    // GPU별 기본 알고리즘 반환
    if (gpu_arch == AMPERE) {
        return ALGO_IMPLICIT_GEMM;  // Ampere 기본
    } else if (gpu_arch == TURING) {
        return ALGO_DIRECT_CONVOLUTION;  // Turing 기본
    } else {
        return ALGO_GEMM;  // 범용 기본
    }
}
```
- **장점**: 빠른 초기화, 안정적
- **단점**: 최적화되지 않은 성능
- **적합한 환경**: 테스트, 디버깅, 호환성 우선

---

## 💾 메모리 관리 인자

### `gpu_mem_limit_gb`

ONNXRuntime이 사용할 수 있는 최대 GPU 메모리를 제한합니다.

```python
# 메모리 할당 계산
total_gpu_memory = 24GB  # RTX A5000
allocation_options = [
    4GB,   # 17% (다른 프로세스와 공유)
    8GB,   # 33% (일반적 권장)
    16GB,  # 67% (고성능)
    21GB,  # 88% (단독 사용)
]
```

**메모리 크기별 특성:**

**4GB (보수적)**
```
장점: 
- 다른 프로세스와 GPU 공유 가능
- 메모리 부족 오류 위험 낮음
- 안정적인 운영

단점:
- 큰 모델이나 배치 처리 제약
- 성능 최적화 제한
```

**8GB (균형)**
```
장점:
- 성능과 안정성의 균형
- 대부분 모델에 충분
- 권장 설정

단점:
- 매우 큰 모델에는 부족할 수 있음
```

**16-21GB (고성능)**
```
장점:
- 최대 성능 발휘 가능
- 대용량 배치 처리 가능
- 메모리 단편화 최소화

단점:
- 다른 프로세스 사용 불가
- 메모리 부족시 시스템 불안정
```

### `arena_extend_strategy`

GPU 메모리 아레나(연속된 메모리 블록)를 확장하는 전략입니다.

**`"kNextPowerOfTwo"` (2^n 확장)**
```cpp
// CUDA 메모리 할당 의사코드
size_t calculateArenaSize(size_t requested_size) {
    // 요청된 크기보다 큰 2의 거듭제곱으로 확장
    size_t power_of_two = 1;
    while (power_of_two < requested_size) {
        power_of_two *= 2;
    }
    return power_of_two;
}

// 예시: 640MB 요청 → 1024MB 할당
```
- **장점**: 메모리 단편화 최소화, 재할당 빈도 감소
- **단점**: 메모리 낭비 가능성
- **적합**: 일정한 크기의 반복 처리

**`"kSameAsRequested"` (요청 크기만큼)**
```cpp
size_t calculateArenaSize(size_t requested_size) {
    return requested_size;  // 정확히 요청된 크기만 할당
}

// 예시: 640MB 요청 → 640MB 할당
```
- **장점**: 메모리 절약, 정확한 할당
- **단점**: 재할당 빈도 증가, 단편화 가능
- **적합**: 가변 크기 처리, 메모리 제약 환경

---

## ⚡ 성능 최적화 인자

### `do_copy_in_default_stream`

CPU-GPU 간 데이터 복사를 어느 CUDA 스트림에서 수행할지 결정합니다.

**`false` (별도 스트림 사용) - 권장**
```cpp
// CUDA 스트림 관리 의사코드
cudaStream_t main_stream;      // 추론용 스트림
cudaStream_t copy_stream;      // 데이터 복사용 스트림

// 병렬 처리 가능
cudaMemcpyAsync(gpu_input, cpu_input, size, copy_stream);  // 복사
kernel_inference<<<blocks, threads, 0, main_stream>>>();   // 추론

// 스트림 동기화
cudaStreamSynchronize(copy_stream);
cudaStreamSynchronize(main_stream);
```
- **장점**: 데이터 복사와 추론 병렬 처리, 처리량 향상
- **단점**: 복잡한 동기화, 디버깅 어려움
- **성능 향상**: 5-15% (데이터 크기에 따라)

**`true` (기본 스트림 사용)**
```cpp
// 순차 처리
cudaMemcpy(gpu_input, cpu_input, size);  // 복사 완료 대기
kernel_inference<<<blocks, threads>>>();  // 추론 시작
cudaDeviceSynchronize();                  // 완료 대기
```
- **장점**: 단순한 구조, 디버깅 용이
- **단점**: 순차 처리로 인한 성능 손실
- **성능**: 기본 수준

### `cudnn_conv_use_max_workspace`

cuDNN이 컨볼루션 연산에 사용할 수 있는 임시 작업공간 크기를 결정합니다.

**`true` (최대 작업공간) - 성능 우선**
```cpp
// cuDNN 작업공간 할당 의사코드
size_t getWorkspaceSize(ConvolutionDesc desc) {
    if (use_max_workspace) {
        // 사용 가능한 최대 메모리의 80% 사용
        return available_gpu_memory * 0.8;
    } else {
        // 최소 필요한 메모리만 사용
        return minimum_required_memory;
    }
}
```
- **장점**: 최고 성능 알고리즘 사용 가능, FFT/Winograd 활성화
- **단점**: 메모리 사용량 증가, OOM 위험
- **성능 향상**: 10-30% (알고리즘 종류에 따라)

**`false` (최소 작업공간) - 메모리 절약**
- **장점**: 메모리 절약, 안정성
- **단점**: 고성능 알고리즘 제한
- **성능**: 기본 수준

### `tunable_op_enable` & `tunable_op_tuning_enable`

런타임에 GPU별 특성에 맞게 연산을 동적으로 최적화하는 기능입니다.

**동작 원리:**
```python
# TunableOp 의사코드
class TunableConvolution:
    def __init__(self):
        self.algorithms = [GEMM, DIRECT, WINOGRAD, FFT]
        self.performance_cache = {}
    
    def execute(self, input_tensor):
        key = (input_tensor.shape, self.filter.shape)
        
        if key not in self.performance_cache:
            # 첫 실행시: 모든 알고리즘 테스트
            best_algo = self.find_best_algorithm(input_tensor)
            self.performance_cache[key] = best_algo
        
        # 캐시된 최적 알고리즘 사용
        return self.performance_cache[key].execute(input_tensor)
```

**`tunable_op_enable: true`**
- **기능**: 런타임 알고리즘 선택 활성화
- **오버헤드**: 첫 실행시 추가 시간 (100-500ms)
- **이후 성능**: 캐시된 최적 알고리즘으로 향상

**`tunable_op_tuning_enable: true`**  
- **기능**: 지속적인 성능 튜닝
- **동작**: 실행 중 더 나은 알고리즘 탐색
- **적응성**: 입력 패턴 변화에 대응

---

## 🏗️ 실행 모드 인자

### `execution_mode`

ONNXRuntime이 그래프의 노드들을 실행하는 방식을 결정합니다.

**`"ORT_SEQUENTIAL"` (순차 실행) - 권장**
```cpp
// 순차 실행 의사코드
void executeGraph(Graph graph) {
    for (Node node : graph.getTopologicalOrder()) {
        // 노드를 하나씩 순서대로 실행
        executeNode(node);
        waitForCompletion(node);  // 완료 대기
    }
}
```
- **특징**: 노드를 하나씩 순서대로 실행
- **장점**: 메모리 사용량 예측 가능, 디버깅 용이, 안정적
- **단점**: 병렬성 제한
- **적합**: 대부분의 추론 작업, 메모리 제약 환경

**`"ORT_PARALLEL"` (병렬 실행)**
```cpp
// 병렬 실행 의사코드
void executeGraph(Graph graph) {
    ThreadPool thread_pool;
    
    for (Node node : graph.getNodes()) {
        if (node.canExecuteInParallel()) {
            thread_pool.submit([&]() {
                executeNode(node);  // 병렬 실행
            });
        } else {
            executeNode(node);  // 순차 실행
        }
    }
    
    thread_pool.waitAll();  // 모든 노드 완료 대기
}
```
- **특징**: 독립적인 노드들을 병렬로 실행
- **장점**: 높은 처리량, GPU 활용률 향상
- **단점**: 메모리 사용량 증가, 동기화 복잡
- **적합**: 복잡한 그래프, 고성능 GPU

### `graph_optimization_level`

ONNX 그래프에 적용할 최적화 수준을 결정합니다.

**`"ORT_ENABLE_ALL"` (모든 최적화) - 권장**
```cpp
// 그래프 최적화 적용 의사코드
Graph optimizeGraph(Graph original_graph) {
    Graph optimized = original_graph;
    
    // 1. 노드 융합 (Node Fusion)
    optimized = fuseConvBatchNormRelu(optimized);
    optimized = fuseMatMulAdd(optimized);
    
    // 2. 상수 접기 (Constant Folding)
    optimized = foldConstants(optimized);
    
    // 3. 불필요한 연산 제거
    optimized = eliminateDeadNodes(optimized);
    
    // 4. 연산 재배치 (Operator Reordering)
    optimized = reorderForMemoryEfficiency(optimized);
    
    // 5. 메모리 최적화
    optimized = optimizeMemoryLayout(optimized);
    
    return optimized;
}
```
- **최적화 종류**: 노드 융합, 상수 접기, 데드 코드 제거, 연산 재배치
- **성능 향상**: 10-40% (모델 복잡도에 따라)
- **메모리 절약**: 중간 결과 제거로 메모리 효율성 향상

**`"ORT_ENABLE_BASIC"` (기본 최적화)**
- **최적화**: 기본적인 융합과 접기만 수행
- **장점**: 빠른 초기화, 호환성
- **단점**: 제한된 성능 향상

---

## 💾 메모리 최적화 인자

### `enable_cpu_mem_arena`

CPU 메모리 아레나를 사용하여 메모리 할당을 최적화합니다.

**`true` (아레나 사용) - 권장**
```cpp
// 메모리 아레나 의사코드
class MemoryArena {
    char* large_buffer;      // 큰 연속 메모리 블록
    size_t current_offset;   // 현재 할당 위치
    
public:
    void* allocate(size_t size) {
        void* ptr = large_buffer + current_offset;
        current_offset += align(size);
        return ptr;  // 매우 빠른 할당
    }
    
    void reset() {
        current_offset = 0;  // 전체 메모리 재사용
    }
};
```
- **장점**: 빠른 메모리 할당/해제, 단편화 방지
- **단점**: 초기 큰 메모리 블록 필요
- **성능 향상**: 5-10% (메모리 집약적 모델)

### `enable_mem_pattern`

메모리 사용 패턴을 분석하여 재사용을 최적화합니다.

**`true` (패턴 최적화) - 권장**
```cpp
// 메모리 패턴 최적화 의사코드
class MemoryPatternOptimizer {
    map<TensorShape, void*> reusable_buffers;
    
public:
    void* getBuffer(TensorShape shape) {
        if (reusable_buffers.contains(shape)) {
            return reusable_buffers[shape];  // 재사용
        } else {
            void* new_buffer = allocate(shape.size());
            reusable_buffers[shape] = new_buffer;
            return new_buffer;
        }
    }
};
```
- **기능**: 동일한 크기 텐서의 메모리 재사용
- **장점**: 메모리 할당 횟수 감소, 가비지 컬렉션 부담 완화
- **성능 향상**: 3-8% (모델 구조에 따라)

---

## 🔄 14가지 조합 구성 원리

### 조합 생성 로직

```python
# 실제 조합 생성 코드
base_configs = [
    {
        'name': 'HEURISTIC_OPTIMIZED',
        'cudnn_conv_algo_search': 'HEURISTIC',
        'do_copy_in_default_stream': False,      # 성능 우선
        'cudnn_conv_use_max_workspace': True,
        'tunable_op_enable': True,
        'execution_mode': 'ORT_SEQUENTIAL',
    },
    {
        'name': 'EXHAUSTIVE_BASELINE', 
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': False,      # 성능 우선
        'cudnn_conv_use_max_workspace': True,
        'tunable_op_enable': False,              # 탐색과 충돌 방지
        'execution_mode': 'ORT_SEQUENTIAL',
    },
    {
        'name': 'DEFAULT_CONSERVATIVE',
        'cudnn_conv_algo_search': 'DEFAULT', 
        'do_copy_in_default_stream': True,       # 안정성 우선
        'cudnn_conv_use_max_workspace': False,
        'tunable_op_enable': False,
        'execution_mode': 'ORT_SEQUENTIAL',
    }
]

memory_limits = [4, 8, 16, 21]  # GPU 메모리별

# 3개 기본 설정 × 4개 메모리 크기 = 12개 조합
# + 2개 실험적 설정 = 14개 조합
```

### 실험적 조합 (13-14번)

**조합 13: 병렬 실행 테스트**
```yaml
cudnn_conv_algo_search: "HEURISTIC"
execution_mode: "ORT_PARALLEL"        # 병렬 모드
gpu_mem_limit_gb: 21                  # 최대 메모리
enable_cpu_mem_arena: False           # 병렬에 최적화
```

**조합 14: 메모리 절약 모드**
```yaml
cudnn_conv_algo_search: "HEURISTIC"
gpu_mem_limit_gb: 4                   # 최소 메모리
do_copy_in_default_stream: True       # 단순 모드
cudnn_conv_use_max_workspace: False   # 메모리 절약
```

## 📊 인자별 성능 영향도

| 인자 | 성능 영향 | 메모리 영향 | 안정성 영향 | 권장도 |
|------|----------|-------------|-------------|---------|
| `cudnn_conv_algo_search` | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | HEURISTIC |
| `do_copy_in_default_stream` | ⭐⭐⭐⭐ | ⭐ | ⭐⭐ | False |
| `gpu_mem_limit_gb` | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 8GB |
| `cudnn_conv_use_max_workspace` | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | True |
| `execution_mode` | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | SEQUENTIAL |
| `tunable_op_enable` | ⭐⭐ | ⭐ | ⭐⭐⭐ | True |

## 💡 실제 사용 권장사항

### 새로운 GPU 환경 최적화 절차

1. **자동 최적화 실행**
   ```bash
   python tools/onnx_optimizer.py --model model.onnx --config config.yaml
   ```

2. **결과 검증**
   ```bash
   # 최적화 전후 성능 비교
   python test_rtmo_performance.py
   ```

3. **프로덕션 적용**
   ```bash
   # config.yaml이 자동 업데이트됨
   # 바로 프로덕션에서 사용 가능
   ```

### GPU별 예상 최적 설정

**고성능 GPU (RTX 4090, A100)**
```yaml
cudnn_conv_algo_search: "HEURISTIC"
gpu_mem_limit_gb: 16-24
do_copy_in_default_stream: false
cudnn_conv_use_max_workspace: true
tunable_op_enable: true
```

**중급 GPU (RTX 3080, A5000)**
```yaml
cudnn_conv_algo_search: "HEURISTIC"  
gpu_mem_limit_gb: 8-12
do_copy_in_default_stream: false
cudnn_conv_use_max_workspace: true
tunable_op_enable: true
```

**엔트리 GPU (RTX 3060, T4)**
```yaml
cudnn_conv_algo_search: "DEFAULT"
gpu_mem_limit_gb: 4-6
do_copy_in_default_stream: true
cudnn_conv_use_max_workspace: false
tunable_op_enable: false
```

---

**이 문서는 자동 최적화 도구와 함께 사용하여 모든 GPU 환경에서 최적 성능을 달성할 수 있도록 설계되었습니다.**