ARG PYTORCH="2.3.1"
ARG CUDA="12.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX 9.0"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV DEBIAN_FRONTEND=noninteractive

# fetch the key refer to https://forums.developer.nvidia.com/t/18-04-cuda-docker-image-is-broken/212892/9
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub 32
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 ffmpeg \
    && vim git lsof \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV
RUN pip install openmim==0.3.9 moviepy==1.0.3 tensorboard
RUN mim install mmengine==0.10.4 mmdet==3.2.0 mmpose==1.3.2
# 컨테이너 생성 시 mmcv 2.1.0 버전이 설치되어 있는 상태
# mmcv의 nms 부분에서 cuda:0 인식이 실패함
# 컨테이너 내부에서 pip install mmcv-full==1.7.2를 설치하면 해당오류가 사라짐
# 도커파일에 mmcv-full==1.7.2 입력하여 이미지 생성을 할 경우 컨테이너 생서 시 설치는 확인되지만, 오류가 동일하게 발생함

# Install MMAction2
RUN conda clean --all
RUN git clone https://github.com/open-mmlab/mmaction2.git /mmaction2
WORKDIR /mmaction2
RUN mkdir -p /mmaction2/data
ENV FORCE_CUDA="1"
ENV MMCV_WITH_OPS=1
RUN git checkout main
RUN pip install cython --no-cache-dir
RUN pip install --no-cache-dir -e .
