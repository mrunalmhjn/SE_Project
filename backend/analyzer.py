import os
import cv2
import numpy as np
from backend.visual import process_visual
from backend.text import process_text
from backend.fusion import fuse_results

def analyze_video(video_path):
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    duration = total / fps if fps > 0 else 0

    # ---- VISUAL ----
    visual_data = process_visual(video_path)

    # ---- TEXT ----
    text_data = process_text(video_path)

    # ---- FUSION ----
    final = fuse_results(visual_data, text_data)

    return {
        "video": os.path.basename(video_path),
        "duration": round(duration, 2),
        "visual": visual_data,
        "text": text_data,
        "final": final
    }