import whisper
from moviepy.editor import VideoFileClip
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import os

model = whisper.load_model("base")
vader = SentimentIntensityAnalyzer()

def extract_audio(video_path):
    audio_path = video_path + ".wav"
    clip = VideoFileClip(video_path)

    if clip.audio is None:
        return None

    clip.audio.write_audiofile(audio_path, fps=16000, logger=None)
    return audio_path


def process_text(video_path):
    audio = extract_audio(video_path)

    if audio is None:
        return {"sentiment": 0, "virality": 0, "text": ""}

    result = model.transcribe(audio)
    text = result["text"]

    v = vader.polarity_scores(text)
    tb = TextBlob(text)

    sentiment = v["compound"] * 0.6 + tb.sentiment.polarity * 0.4

    virality = abs(sentiment) + tb.sentiment.subjectivity

    return {
        "text": text[:300],
        "sentiment": sentiment,
        "virality": min(virality, 1)
    }