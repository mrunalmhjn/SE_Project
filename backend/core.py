import cv2, numpy as np, torch, os, warnings
import torchvision.models as models
import torchvision.transforms as transforms
from skimage.feature import local_binary_pattern, hog
from skimage.filters import gabor
from skimage import color
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
from tabulate import tabulate

import whisper
from moviepy.editor import VideoFileClip
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

warnings.filterwarnings('ignore')


def run_full_analysis(VIDEO_PATH):

    os.makedirs("outputs", exist_ok=True)

    # ================= VIDEO INFO =================
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    duration = total / fps if fps > 0 else 0

    # ================= FRAME SAMPLING =================
    def sample_frames(video_path, sample_every_n=15, max_frames=60):
        cap = cv2.VideoCapture(video_path)
        frames, idx = [], 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if idx % sample_every_n == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            idx += 1
            if len(frames) >= max_frames: break
        cap.release()
        return frames

    frames = sample_frames(VIDEO_PATH)

    # ================= VISUAL SCORES =================
    def visual_sentiment_score(img):
        hsv = color.rgb2hsv(img)
        return (hsv[:,:,2].mean() - 0.5) + hsv[:,:,1].mean()

    def visual_virality_score(img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return np.clip(cv2.Laplacian(gray, cv2.CV_64F).var()/2000,0,1)

    vis_sent = [visual_sentiment_score(f) for f in frames]
    vis_vir  = [visual_virality_score(f) for f in frames]

    # ================= AUDIO + TEXT =================
    AUDIO_PATH = VIDEO_PATH + "_audio.wav"

    def extract_audio(video_path):
        clip = VideoFileClip(video_path)
        if clip.audio is None:
            return False
        clip.audio.write_audiofile(AUDIO_PATH, fps=16000, logger=None)
        return True

    has_audio = extract_audio(VIDEO_PATH)

    transcript_text = ""
    if has_audio:
        model = whisper.load_model("base")
        result = model.transcribe(AUDIO_PATH)
        transcript_text = result["text"]

    vader = SentimentIntensityAnalyzer()

    def text_sentiment(text):
        if not text.strip():
            return 0
        v = vader.polarity_scores(text)
        tb = TextBlob(text)
        return v["compound"]*0.6 + tb.sentiment.polarity*0.4

    text_score = text_sentiment(transcript_text)

    # ================= FUSION =================
    fused_sent = 0.6*text_score + 0.4*np.mean(vis_sent)
    fused_vir  = 0.6*abs(text_score) + 0.4*np.mean(vis_vir)

    # ================= LABELS =================
    sent_label = "Positive" if fused_sent>0.05 else "Negative" if fused_sent<-0.05 else "Neutral"
    vir_label  = "Viral" if fused_vir>0.45 else "Non-Viral"

    # ================= DASHBOARD =================
    plt.figure(figsize=(10,5))
    plt.plot(vis_sent, label="Visual Sentiment")
    plt.plot(vis_vir, label="Visual Virality")
    plt.legend()
    plt.title("Analysis Timeline")
    plt.savefig("outputs/dashboard.png")
    plt.close()

    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 2)

    # 1. Sentiment timeline
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(vis_sent, color='green')
    ax1.set_title("Visual Sentiment Timeline")

    # 2. Virality timeline
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(vis_vir, color='red')
    ax2.set_title("Visual Virality Timeline")

    # 3. Histogram
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(vis_sent, bins=20)
    ax3.set_title("Sentiment Distribution")

    # 4. Virality histogram
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(vis_vir, bins=20)
    ax4.set_title("Virality Distribution")

    # 5. Summary text
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')

    summary_text = f"""
    VIDEO ANALYSIS SUMMARY

    Sentiment: {sent_label} ({fused_sent:.3f})
    Virality: {vir_label} ({fused_vir:.3f})

    Frames analyzed: {len(frames)}
    Duration: {duration:.2f}s
    """

    ax5.text(0.1, 0.5, summary_text, fontsize=12)

    plt.tight_layout()
    plt.savefig("outputs/dashboard.png", dpi=150)
    plt.close()

    # ================= CSV =================
    df = pd.DataFrame({
        "frame": list(range(len(frames))),
        "sentiment": vis_sent,
        "virality": vis_vir
    })
    df.to_csv("outputs/frame_data.csv", index=False)

    summary = pd.DataFrame([{
        "sentiment": fused_sent,
        "sentiment_label": sent_label,
        "virality": fused_vir,
        "virality_label": vir_label
    }])
    summary.to_csv("outputs/summary.csv", index=False)

    return {
        "sentiment": sent_label,
        "virality": vir_label,
        "dashboard": "outputs/dashboard.png"
    }