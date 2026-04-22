import cv2
import numpy as np
from skimage import color

def sample_frames(video_path, step=15, max_frames=60):
    cap = cv2.VideoCapture(video_path)
    frames, i = [], 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % step == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        i += 1
        if len(frames) >= max_frames:
            break

    cap.release()
    return frames


def visual_sentiment(frame):
    hsv = color.rgb2hsv(frame)
    brightness = hsv[:, :, 2].mean()
    saturation = hsv[:, :, 1].mean()

    return (brightness - 0.5) + saturation


def visual_virality(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var() / 1000
    edges = cv2.Canny(gray, 50, 150).mean() / 255.0

    return np.clip(sharpness + edges, 0, 1)


def process_visual(video_path):
    frames = sample_frames(video_path)

    sent = [visual_sentiment(f) for f in frames]
    vir = [visual_virality(f) for f in frames]

    return {
        "frames": len(frames),
        "sentiment": float(np.mean(sent)),
        "virality": float(np.mean(vir))
    }