# main.py - 메인 애플리케이션 (출력 경로 UI 추가 버전)
import argparse
import os
import os.path as osp
from tempfile import TemporaryDirectory
import threading
import time
import queue
import csv
from datetime import datetime

# MMEngine과 MMPose/MMAction2 import 순서를 조정
import mmengine
import mmpose.datasets.transforms
import mmaction.datasets.transforms

import numpy as np
import cv2
import gradio as gr
from sklearn.metrics import confusion_matrix
import seaborn as sns
# Matplotlib 백엔드를 'Agg'로 명시적으로 설정
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 커스텀 모듈 import
from pose_detector import PoseDetector
from action_classifier import ActionClassifier

# Global variables
pose_detector = None
action_classifier = None
processing_queue = queue.Queue()
stop_event = threading.Event()
progress_info = {
    'total_videos': 0, 'processed_videos': 0, 'current_video': '',
    'current_frame': 0, 'total_frames': 0, 'is_processing': False,
    'video_progress': 0.0, 'frame_progress': 0.0
}
frame_lock = threading.Lock()
current_frame_data = {
    'frame': None, 'timestamp': 0, 'prediction': None, 'confidence': 0.0
}
confusion_matrix_data = {
    'predictions': [], 'labels': []
}

# --- CSV 로깅 함수 (출력 디렉토리 인자 추가) ---
def log_batch_result(output_dir, file_name, start_frame, video_time, prediction, gt_result):
    os.makedirs(output_dir, exist_ok=True)
    file_path = osp.join(output_dir, 'batch_results.csv')
    file_exists = osp.exists(file_path)
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['file_name', 'start_frame', 'start_time', 'inference_result', 'gt_result'])
        pred_label = "Fight" if prediction == 1 else "NonFight"
        gt_label = "Fight" if gt_result == 1 else "NonFight"
        writer.writerow([file_name, start_frame, f"{video_time:.2f}s", pred_label, gt_label])

def log_continuity_result(output_dir, file_name, continuity_threshold, all_predictions, gt_result):
    final_result = 0
    if continuity_threshold > 0:
        consecutive_fights = 0
        for pred in all_predictions:
            if pred == 1:
                consecutive_fights += 1
                if consecutive_fights >= continuity_threshold:
                    final_result = 1
                    break
            else:
                consecutive_fights = 0
    
    if gt_result == 1 and final_result == 1: confusion = 'TP'
    elif gt_result == 0 and final_result == 0: confusion = 'TN'
    elif gt_result == 0 and final_result == 1: confusion = 'FP'
    else: confusion = 'FN'

    os.makedirs(output_dir, exist_ok=True)
    file_path = osp.join(output_dir, 'continuity_results.csv')
    file_exists = osp.exists(file_path)
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['file_name', 'continuity_times', 'final_result', 'gt_result', 'confusion'])
        final_label = "Fight" if final_result == 1 else "NonFight"
        gt_label = "Fight" if gt_result == 1 else "NonFight"
        writer.writerow([file_name, continuity_threshold, final_label, gt_label, confusion])

    return final_result, gt_result

def draw_prediction_overlay(image, prediction, confidence, ground_truth=None):
    height, width = image.shape[:2]
    pred_text = "Fight" if prediction == 1 else "NonFight"
    conf_text = f"Confidence: {confidence:.2f}"
    if prediction == 1: color = (255, 0, 0)
    else: color = (0, 255, 0)
    overlay = image.copy()
    cv2.rectangle(overlay, (10, 10), (160, 60), color, -1)
    image = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    cv2.putText(image, f"Pred: {pred_text}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, f"Conf: {conf_text}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    if ground_truth is not None:
        gt_text = "GT: Fight" if ground_truth == 1 else "GT: NonFight"
        cv2.putText(image, gt_text, (width - 120, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    return image

def add_frame_to_buffer(frame, frame_idx, total_frames, prediction=None, confidence=0.0):
    global current_frame_data
    with frame_lock:
        current_frame_data = {'frame': frame, 'frame_idx': frame_idx, 'total_frames': total_frames, 'timestamp': time.time(), 'prediction': prediction, 'confidence': confidence}

def process_video_with_models(vid_path, pose_detector, action_classifier, continuity_threshold, output_dir):
    global progress_info, stop_event, confusion_matrix_data
    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 30.0
    cap.release()

    try:
        with TemporaryDirectory() as tmp_dir:
            frame_paths = action_classifier.extract_frames(vid_path, tmp_dir)
            if not frame_paths or not isinstance(frame_paths[0], list): return
            if isinstance(frame_paths, tuple): frame_paths = frame_paths[0]
            progress_info['total_frames'] = len(frame_paths)
            progress_info['current_frame'] = 0
            ground_truth = 1 if 'Fight' in vid_path else 0
            keypoints_sequence, scores_sequence, video_predictions = [], [], []

            for frame_idx, frame_path in enumerate(frame_paths):
                if stop_event.is_set(): return
                frame = cv2.imread(frame_path)
                if frame is None: continue
                keypoints, scores, _ = pose_detector.detect_pose(frame)
                if keypoints is not None and scores is not None:
                    keypoints_sequence.append(keypoints.copy())
                    scores_sequence.append(scores.copy())
                    prediction, confidence = 0, 0.5
                    if len(keypoints_sequence) >= action_classifier.sequence_length:
                        prediction, confidence = action_classifier.predict(keypoints_sequence, scores_sequence)
                        video_predictions.append(prediction)
                        start_frame = frame_idx - action_classifier.sequence_length + 1
                        video_time = start_frame / fps if fps > 0 else 0
                        log_batch_result(output_dir, osp.basename(vid_path), start_frame, video_time, prediction, ground_truth)
                    frame_with_keypoints = pose_detector.draw_keypoints(frame, keypoints, scores, 0.3)
                    frame_with_overlay = draw_prediction_overlay(frame_with_keypoints, prediction, confidence, ground_truth)
                    add_frame_to_buffer(frame_with_overlay, frame_idx, len(frame_paths), prediction, confidence)
                else:
                    frame_with_overlay = draw_prediction_overlay(frame, 0, 0.5, ground_truth)
                    add_frame_to_buffer(frame_with_overlay, frame_idx, len(frame_paths), 0, 0.5)
                progress_info['current_frame'] = frame_idx + 1
                progress_info['frame_progress'] = (frame_idx + 1) / len(frame_paths) * 100
            
            final_pred, final_gt = log_continuity_result(output_dir, osp.basename(vid_path), continuity_threshold, video_predictions, ground_truth)
            confusion_matrix_data['predictions'].append(final_pred)
            confusion_matrix_data['labels'].append(final_gt)
    except Exception as e:
        print(f"Error processing {vid_path}: {e}")

def init_pose_model(config_path, checkpoint_path, device, nms_thr, score_thr):
    global pose_detector
    if pose_detector is None: pose_detector = PoseDetector()
    _, message = pose_detector.init_model(config_path, checkpoint_path, device, nms_thr, score_thr)
    return message

def init_gcn_model(config_path, checkpoint_path, device, sequence_length, confidence_threshold):
    global action_classifier
    if action_classifier is None: action_classifier = ActionClassifier()
    _, message = action_classifier.init_model(config_path, checkpoint_path, device, sequence_length, confidence_threshold)
    return message

def start_inference(input_path, output_dir, pose_score_threshold, continuity_threshold):
    global progress_info, pose_detector, action_classifier, stop_event, confusion_matrix_data
    if not all([pose_detector, pose_detector.model, action_classifier]): return "Initialize both models first!"
    if not input_path: return "Input Path cannot be empty!"
    if not output_dir: return "Output Directory cannot be empty!"

    pose_detector.score_thr = pose_score_threshold
    progress_info['is_processing'] = True
    stop_event.clear()
    confusion_matrix_data = {'predictions': [], 'labels': []}

    def process_thread():
        video_files = []
        if osp.isdir(input_path):
            video_files.extend([osp.join(root, file) for root, _, files in os.walk(input_path) for file in files if file.endswith(('.avi', '.mp4', '.mov', '.mkv'))])
        elif osp.isfile(input_path): video_files.append(input_path)
        else:
            processing_queue.put(('error', f"Invalid input path: {input_path}"))
            return

        progress_info['total_videos'] = len(video_files)
        progress_info['processed_videos'] = 0
        for vid_path in video_files:
            if stop_event.is_set():
                processing_queue.put(('message', "Processing stopped by user"))
                break
            progress_info['current_video'] = osp.basename(vid_path)
            processing_queue.put(('message', f"Processing: {osp.basename(vid_path)}..."))
            process_video_with_models(vid_path, pose_detector, action_classifier, continuity_threshold, output_dir)
            if not stop_event.is_set():
                progress_info['processed_videos'] += 1
                progress_info['video_progress'] = (progress_info['processed_videos'] / progress_info['total_videos']) * 100
                processing_queue.put(('message', f"Completed: {osp.basename(vid_path)}"))
        progress_info['is_processing'] = False
        processing_queue.put(('message', "Processing completed!" if not stop_event.is_set() else "Processing stopped!"))

    thread = threading.Thread(target=process_thread)
    thread.daemon = True
    thread.start()
    return "Inference started!"

def stop_processing():
    global stop_event
    stop_event.set()
    return "Stop signal sent!", gr.Timer(active=False)

def generate_confusion_matrix():
    global confusion_matrix_data
    fig, ax = plt.subplots(figsize=(4.5, 3))
    
    y_true = confusion_matrix_data['labels']
    y_pred = confusion_matrix_data['predictions']

    if not y_true:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        # 데이터가 없을 때도 2x2 회색 매트릭스 틀을 보여줌
        cm_empty = np.zeros((2, 2))
        sns.heatmap(cm_empty, annot=False, fmt='d', cmap='Greys', cbar=False, ax=ax, xticklabels=['NonFight', 'Fight'], yticklabels=['NonFight', 'Fight'])

    else:
        # labels=[0, 1] 인자를 추가하여 항상 2x2 행렬을 생성하도록 강제
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['NonFight', 'Fight'], yticklabels=['NonFight', 'Fight'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    ax.set_title('Confusion Matrix (Video-level)', fontsize=10)
    plt.tight_layout()
    return fig

def get_updates():
    global current_frame_data, progress_info, frame_lock
    messages = []
    try:
        while not processing_queue.empty():
            msg_type, data = processing_queue.get_nowait()
            if msg_type == 'message': messages.append(data)
    except queue.Empty: pass

    with frame_lock:
        latest_frame = current_frame_data.get('frame')
        prediction = current_frame_data.get('prediction')
        confidence = current_frame_data.get('confidence')

    video_progress = round(progress_info.get('video_progress', 0.0), 1)
    frame_progress = round(progress_info.get('frame_progress', 0.0), 1)
    video_status = f"Videos: {progress_info['processed_videos']}/{progress_info['total_videos']} ({video_progress:.1f}%)"
    frame_status = f"Current: {progress_info['current_video']} - {progress_info['current_frame']}/{progress_info['total_frames']} ({frame_progress:.1f}%)"
    log_text = "\n".join(messages[-10:]) if messages else "Waiting for logs..."

    inference_result_text = "Status: Idle"
    if progress_info['is_processing']:
        pred_label = "Fight" if prediction == 1 else "NonFight"
        inference_result_text = f"Prediction: {pred_label}\nConfidence: {confidence:.3f}"

    if latest_frame is None:
        latest_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    else:
        latest_frame = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB)
        latest_frame = cv2.resize(latest_frame, (640, 360))

    return latest_frame, inference_result_text, video_status, frame_status, log_text, video_progress, frame_progress, generate_confusion_matrix()

def create_gradio_interface():
    with gr.Blocks(title="Violence Detection", theme=gr.themes.Default()) as demo:
        gr.Markdown("## Real-time Violence Detection using RTMO and GCN")

        with gr.Tab("Model Settings"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Pose Model Configuration")
                    pose_config_path = gr.Textbox(label="Pose Model Config Path", \
                        value="/workspace/mmpose/configs/body_2d_keypoint/rtmo/body7/rtmo-m_16xb16-600e_body7-640x640.py")
                    pose_checkpoint_path = gr.Textbox(label="Pose Model Checkpoint Path", \
                        value="/workspace/mmpose/checkpoints/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.pth")
                    pose_device = gr.Dropdown(choices=["cuda:0", "cuda:1", "cpu"], value="cuda:0", label="Pose Device")
                    nms_thr_slider = gr.Slider(0.0, 1.0, 0.5, step=0.01, label="NMS Threshold")
                    score_thr_slider = gr.Slider(0.0, 1.0, 0.2, step=0.01, label="Score Threshold")
                    init_pose_btn = gr.Button("Initialize Pose Model")
                    pose_init_status = gr.Textbox(label="Pose Model Status", interactive=False)
                with gr.Column():
                    gr.Markdown("### GCN Model Configuration")
                    gcn_config_path = gr.Textbox(label="GCN Config Path", \
                        value="/workspace/mmaction2/configs/skeleton/stgcnpp/stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d_rwf2000_finetune_0.py")
                    gcn_checkpoint_path = gr.Textbox(label="GCN Checkpoint Path", \
                        value="/workspace/mmaction2/work_dirs/stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d_rwf2000_finetune_0/best_acc_top1_epoch_23.pth")
                    gcn_device = gr.Dropdown(choices=["cuda:0", "cuda:1", "cpu"], value="cuda:0", label="GCN Device")
                    sequence_length = gr.Slider(10, 100, 30, step=1, label="Sequence Length")
                    confidence_threshold = gr.Slider(0.0, 1.0, 0.5, step=0.01, label="Confidence Threshold")
                    init_gcn_btn = gr.Button("Initialize GCN Model")
                    gcn_init_status = gr.Textbox(label="GCN Model Status", interactive=False)

        with gr.Tab("Inference"):
            timer = gr.Timer(0.04, active=True)
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### Input & Control")
                    input_path = gr.Textbox(label="Input Path (Video file or folder)", placeholder="e.g., C:/videos or C:/videos/fight.mp4", \
                                            value="/aivanas/raw/surveillance/action/violence/action_recognition/data/UCF_Crime/anomaly_videos/Anomaly-Videos-Part-2/Anomaly-Videos-Part-2/Fighting")
                    output_dir = gr.Textbox(label="Output Directory for CSV results", value="/workspace/rtmo_gcn_test")
                    pose_score_threshold = gr.Slider(0.0, 1.0, 0.3, step=0.01, label="Pose Score Threshold")
                    continuity_threshold = gr.Slider(1, 10, 3, step=1, label="Continuity Times (for Final Result)")
                    with gr.Row():
                        start_btn = gr.Button("Start Inference", variant="primary")
                        stop_btn = gr.Button("Stop", variant="stop")
                    gr.Markdown("### Progress & Logs")
                    video_progress_bar = gr.Slider(label="Video Progress (%)", interactive=False)
                    frame_progress_bar = gr.Slider(label="Frame Progress (%)", interactive=False)
                    video_status = gr.Textbox(label="Video Status", interactive=False)
                    frame_status = gr.Textbox(label="Frame Status", interactive=False)
                    log_output = gr.Textbox(label="Log", lines=5, interactive=False, max_lines=10)

                with gr.Column(scale=3):
                    gr.Markdown("### Live Output")
                    current_frame_display = gr.Image(label="Real-time Display", height=360, width=640)
                    with gr.Row():
                        inference_result_output = gr.Textbox(label="Inference Result (per frame-seq)", interactive=False, lines=2, scale=1)
                        confusion_matrix_plot = gr.Plot(label="Confusion Matrix (per video)", scale=2)

        # Event Handlers
        init_pose_btn.click(init_pose_model, [pose_config_path, pose_checkpoint_path, pose_device, nms_thr_slider, score_thr_slider], [pose_init_status])
        init_gcn_btn.click(init_gcn_model, [gcn_config_path, gcn_checkpoint_path, gcn_device, sequence_length, confidence_threshold], [gcn_init_status])
        start_btn.click(start_inference, [input_path, output_dir, pose_score_threshold, continuity_threshold], [log_output]).then(lambda: gr.Timer(active=True), outputs=[timer])
        stop_btn.click(stop_processing, outputs=[log_output, timer])
        timer.tick(get_updates, outputs=[current_frame_display, inference_result_output, video_status, frame_status, log_output, video_progress_bar, frame_progress_bar, confusion_matrix_plot])
    return demo

def main():
    parser = argparse.ArgumentParser(description='Real-time Violence Detection with CSV Logging')
    parser.add_argument('--server-name', default='0.0.0.0', help='Server name')
    parser.add_argument('--server-port', type=int, default=7862, help='Server port')
    parser.add_argument('--share', action='store_true', help='Share the interface')
    args = parser.parse_args()

    demo = create_gradio_interface()
    print(f"Starting Violence Detection Interface on: {args.server_name}:{args.server_port}")
    demo.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)

if __name__ == "__main__":
    main()