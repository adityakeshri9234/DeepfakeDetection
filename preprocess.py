import os
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from skimage.transform import resize
import random

# Function to compute optical flow
def compute_optical_flow(prev_frame, next_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

# Function to process a single video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    processed_frames = []
    success, prev_frame = cap.read()
    if not success:
        return None

    for _ in range(1, frame_count):
        success, next_frame = cap.read()
        if not success:
            break

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces1 = face_cascade.detectMultiScale(prev_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        faces2 = face_cascade.detectMultiScale(next_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces1) > 0 and len(faces2) > 0:
            (x1, y1, w1, h1) = faces1[0]
            (x2, y2, w2, h2) = faces2[0]
            face1 = prev_frame[y1:y1+h1, x1:x1+w1]
            face2 = next_frame[y2:y2+h2, x2:x2+w2]
            face1 = resize(face1, (224, 224))
            face2 = resize(face2, (224, 224))

            # Ensure the frames are in uint8 format
            face1 = (face1 * 255).astype(np.uint8)
            face2 = (face2 * 255).astype(np.uint8)

            optical_flow = compute_optical_flow(face1, face2)
            processed_frames.append(optical_flow)

        prev_frame = next_frame

    cap.release()

    if len(processed_frames) == 0:
        return None

    return processed_frames

# Function to save frames to disk
def save_frames(frames, output_dir, label, video_name):
    os.makedirs(output_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        frame_path = os.path.join(output_dir, f"{label}_{video_name}_{i}.npy")
        np.save(frame_path, frame)

# Function to process and save a single video
def process_and_save_video(video_info):
    video_path, label, output_dir = video_info
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frames = process_video(video_path)
    if frames is not None:
        save_frames(frames, output_dir, label, video_name)
    return video_path

# Function to create dataset with multiprocessing
def create_dataset(data_dir, output_dir, train_ratio=0.8, seed=42, num_workers=256):
    random.seed(seed)
    subdirs = {
        'Celeb-real': 'real',
        'Celeb-synthesis': 'fake',
        'YouTube-real': 'real'
    }
    video_paths = []
    for subdir, label in subdirs.items():
        video_dir = os.path.join(data_dir, subdir)
        video_paths += [(os.path.join(video_dir, video), label, os.path.join(output_dir, label)) for video in os.listdir(video_dir) if video.endswith('.mp4')]

    random.shuffle(video_paths)
    split_idx = int(len(video_paths) * train_ratio)
    train_videos = video_paths[:split_idx]
    test_videos = video_paths[split_idx:]
    cnt = 1
    for split, videos in [('train', train_videos), ('test', test_videos)]:
        for subdir, label in subdirs.items():
            split_label_dir = os.path.join(output_dir, split, label)
            os.makedirs(split_label_dir, exist_ok=True)

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_and_save_video, (video_path, label, os.path.join(output_dir, split, label))) for (video_path, label, _) in videos]
            for future in as_completed(futures):
                try:
                    video_path = future.result()
                    print(f"{cnt} Processed {video_path}")
                    cnt += 1
                except Exception as e:
                    print(f"Error processing video: {e}")

data_dir = '/kaggle/input/celeb-v1-df'  # Replace with your dataset path
output_dir = 'processed_celeb_df'  # Replace with the desired output path

create_dataset(data_dir, output_dir)
print(f'Dataset processed and saved in {output_dir}')
