# Copyright (c) OpenMMLab. All rights reserved.
import logging  # 로깅 모듈 임포트
import mimetypes  # 파일 타입 추론을 위한 모듈 임포트
import os  # OS 관련 기능 사용을 위한 모듈 임포트
import time  # 시간 관련 함수 사용을 위한 모듈 임포트
from argparse import ArgumentParser  # 명령행 인자 파싱을 위한 모듈 임포트

import cv2  # OpenCV: 이미지/비디오 처리 라이브러리
import json_tricks as json  # JSON 입출력 (numpy 지원) 라이브러리
import mmcv  # OpenMMLab의 컴퓨터 비전 유틸리티
import mmengine  # OpenMMLab의 엔진 유틸리티
import numpy as np  # 수치 계산을 위한 numpy
from mmengine.logging import print_log  # mmengine의 로그 출력 함수

from mmpose.apis import inference_bottomup, init_model  # mmpose의 추론 및 모델 초기화 함수 임포트
from mmpose.registry import VISUALIZERS  # 시각화 레지스트리 임포트
from mmpose.structures import split_instances  # 인스턴스 분리 함수 임포트


def process_one_image(args,
                      img,
                      pose_estimator,
                      visualizer=None,
                      show_interval=0):
    """한 장의 이미지에 대해 예측된 keypoint(및 heatmap)를 시각화하는 함수"""

    # 단일 이미지에 대해 추론 수행
    batch_results = inference_bottomup(pose_estimator, img)  # bottom-up 추론 실행
    results = batch_results[0]  # 첫 번째 결과만 사용

    # 결과를 시각화할 이미지 준비
    if isinstance(img, str):  # 이미지가 파일 경로인 경우
        img = mmcv.imread(img, channel_order='rgb')  # 이미지를 RGB로 읽기
    elif isinstance(img, np.ndarray):  # 이미지가 numpy 배열인 경우
        img = mmcv.bgr2rgb(img)  # BGR을 RGB로 변환

    if visualizer is not None:  # 시각화기가 있을 때
        visualizer.add_datasample(
            'result',  # 데이터 샘플 이름
            img,  # 시각화할 이미지
            data_sample=results,  # 예측 결과
            draw_gt=False,  # GT(정답) 미표시
            draw_bbox=True,  # 바운딩 박스 미표시
            draw_heatmap=args.draw_heatmap,  # heatmap 시각화 여부
            show_kpt_idx=args.show_kpt_idx,  # keypoint 인덱스 표시 여부
            show=args.show,  # 화면에 표시 여부
            wait_time=show_interval,  # 프레임 간 대기 시간
            kpt_thr=args.kpt_thr)  # keypoint 점수 임계값

    return results.pred_instances  # 예측된 인스턴스 반환


def parse_args():
    parser = ArgumentParser()  # 명령행 인자 파서 생성
    parser.add_argument('config', help='Config file')  # config 파일 경로
    parser.add_argument('checkpoint', help='Checkpoint file')  # 체크포인트 파일 경로
    parser.add_argument(
        '--input', type=str, 
        default='/aivanas/raw/surveillance/action/violence/action_recognition/data/RWF-2000', help='Image/Video file')  # 입력 파일 경로
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')  # 결과를 화면에 표시할지 여부
    parser.add_argument(
        '--output-root',
        type=str,
        default='/workspace/mmpose/output',
        help='root of the output img file. '
        'Default not saving the visualization images.')  # 결과 저장 폴더
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        default=True,
        help='whether to save predicted results')  # 예측값 저장 여부
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')  # 추론 디바이스
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        help='Visualize the predicted heatmap')  # heatmap 시각화 여부
    parser.add_argument(
        '--show-kpt-idx',
        action='store_true',
        default=False,
        help='Whether to show the index of keypoints')  # keypoint 인덱스 표시 여부
    parser.add_argument(
        '--kpt-thr', type=float, default=0.9, help='Keypoint score threshold')  # keypoint 점수 임계값
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization')  # keypoint 시각화 반지름
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')  # 연결선 두께
    parser.add_argument(
        '--show-interval', type=int, default=0, help='Sleep seconds per frame')  # 프레임 간 대기 시간
    args = parser.parse_args()  # 인자 파싱
    return args  # 파싱된 인자 반환


def main():
    args = parse_args()  # 인자 파싱
    assert args.show or (args.output_root != '')  # 결과 표시 또는 저장 중 하나는 필수
    assert args.input != ''  # 입력 파일 필수

    # config 및 checkpoint로 모델 생성
    if args.draw_heatmap:
        cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))  # heatmap 출력 옵션 추가
    else:
        cfg_options = None  # 옵션 없음

    model = init_model(
        args.config,
        args.checkpoint,
        device=args.device,
        cfg_options=cfg_options)  # 모델 초기화

    # 시각화기 생성
    model.cfg.visualizer.radius = args.radius  # keypoint 반지름 설정
    model.cfg.visualizer.line_width = args.thickness  # 연결선 두께 설정
    visualizer = VISUALIZERS.build(model.cfg.visualizer)  # 시각화기 빌드
    visualizer.set_dataset_meta(model.dataset_meta)  # 데이터셋 메타정보 설정

    def process_video_file(video_path):
        output_file = None
        if args.output_root:
            mmengine.mkdir_or_exist(args.output_root)
            output_file = os.path.join(args.output_root, os.path.basename(video_path))
            if video_path == 'webcam':
                output_file += '.mp4'

        if args.save_predictions:
            assert args.output_root != ''
            pred_save_path = f'{args.output_root}/results_' \
                f'{os.path.splitext(os.path.basename(video_path))[0]}.json'

        input_type = 'webcam' if video_path == 'webcam' else mimetypes.guess_type(video_path)[0].split('/')[0]

        if input_type == 'image':
            pred_instances = process_one_image(
                args, video_path, model, visualizer, show_interval=0)
            if args.save_predictions:
                pred_instances_list = split_instances(pred_instances)
            if output_file:
                img_vis = visualizer.get_image()
                mmcv.imwrite(mmcv.rgb2bgr(img_vis), output_file)
            if args.save_predictions:
                with open(pred_save_path, 'w') as f:
                    json.dump(
                        dict(
                            meta_info=model.dataset_meta,
                            instance_info=pred_instances_list),
                        f,
                        indent='\t')
                print(f'predictions have been saved at {pred_save_path}')
            if output_file:
                print_log(
                    f'the output {input_type} has been saved at {output_file}',
                    logger='current',
                    level=logging.INFO)

        elif input_type in ['webcam', 'video']:
            if video_path == 'webcam':
                cap = cv2.VideoCapture(0)
            else:
                cap = cv2.VideoCapture(video_path)
            video_writer = None
            pred_instances_list = []
            frame_idx = 0
            while cap.isOpened():
                success, frame = cap.read()
                frame_idx += 1
                if not success:
                    break
                pred_instances = process_one_image(args, frame, model, visualizer, 0.001)
                if args.save_predictions:
                    pred_instances_list.append(
                        dict(
                            frame_id=frame_idx,
                            instances=split_instances(pred_instances)))
                if output_file:
                    frame_vis = visualizer.get_image()
                    if video_writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(
                            output_file,
                            fourcc,
                            25,
                            (frame_vis.shape[1], frame_vis.shape[0]))
                    video_writer.write(mmcv.rgb2bgr(frame_vis))
                if args.show:
                    if cv2.waitKey(5) & 0xFF == 27:
                        break
                    time.sleep(args.show_interval)
            if video_writer:
                video_writer.release()
            cap.release()
            if args.save_predictions:
                with open(pred_save_path, 'w') as f:
                    json.dump(
                        dict(
                            meta_info=model.dataset_meta,
                            instance_info=pred_instances_list),
                        f,
                        indent='\t')
                print(f'predictions have been saved at {pred_save_path}')
            if output_file:
                print_log(
                    f'the output {input_type} has been saved at {output_file}',
                    logger='current',
                    level=logging.INFO)
        else:
            print(f"file {os.path.basename(video_path)} has invalid format. 건너뜀.")

    # 입력이 폴더인지 파일인지 분기
    if os.path.isdir(args.input):
        # 폴더 내 모든 비디오 파일 반복
        video_exts = ('.mp4', '.avi', '.mov', '.mkv')
        video_files = []
        for root, _, files in os.walk(args.input):
            for f in files:
                if f.lower().endswith(video_exts):
                    video_files.append(os.path.join(root, f))
        if not video_files:
            print(f"폴더 내에 비디오 파일이 없습니다: {args.input}")
            return
        for video_path in video_files:
            print(f"\n==== {os.path.basename(video_path)} 추론 시작 ====")
            process_video_file(video_path)
    else:
        # 기존 단일 파일 처리
        process_video_file(args.input)

if __name__ == '__main__':
    main()  # 메인 함수 실행
