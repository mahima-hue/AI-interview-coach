import streamlit as st
import cv2
import requests
import base64
import numpy as np
import pandas as pd
from collections import Counter

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# ================= CONFIG =================
API_URL = "https://router.huggingface.co/hf-inference/models/trpakov/vit-face-expression"
HEADERS = {"Authorization": "Bearer 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"}

# ================= UI =================
st.set_page_config(page_title="AI Interview Coach", layout="wide")

st.markdown("""
<style>
body {background-color:#0e1117; color:white;}
h1, h2, h3 {text-align:center;}
</style>
""", unsafe_allow_html=True)

st.title("🎤 AI Interview Behavior Analyzer")
st.markdown("<p style='text-align:center;'>Real-time AI Mock Interview System</p>", unsafe_allow_html=True)

# ================= FUNCTIONS =================

def get_emotion(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    encoded = base64.b64encode(buffer.tobytes()).decode("utf-8")

    try:
        res = requests.post(API_URL, headers=HEADERS, json={"inputs": encoded}, timeout=5)
        if res.status_code == 200:
            result = res.json()
            return result[0]['label'], result[0]['score']
    except:
        pass

    return "neutral", 0.5


def detect_posture(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face.detectMultiScale(gray,1.3,5)

    if len(faces)==0:
        return "Not Visible"

    x,y,fw,fh = faces[0]
    center = x + fw//2

    if center < w*0.3:
        return "Looking Left"
    elif center > w*0.7:
        return "Looking Right"
    else:
        return "Centered"


# ================= AI ANALYSIS =================

def generate_report(history, emotions, posture):

    avg = np.mean(history)
    std = np.std(history)

    trend = "improving" if history[-1] > history[0] else "declining" if history[-1] < history[0] else "stable"

    counts = Counter(emotions)
    dominant = max(counts, key=counts.get)

    posture_main = max(set(posture), key=posture.count)

    return f"""
## 🧠 AI Interview Report

Confidence trend: **{trend}**  
Average confidence: **{round(avg*10)}/10**  
Stability: **{'high' if std<0.1 else 'medium' if std<0.2 else 'low'}**

Emotion: **{dominant}**  
Posture: **{posture_main}**

### 🎯 Recommendations
- Maintain eye contact  
- Improve expressiveness  
- Keep posture centered  
- Practice consistency  

### 🏁 Verdict
{'Strong performance' if avg>0.7 else 'Needs improvement'}
"""


# ================= VIDEO CLASS =================

class Analyzer(VideoTransformerBase):
    def __init__(self):
        self.history = []
        self.emotions = []
        self.posture = []

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        small = cv2.resize(img,(224,224))

        emotion, conf = get_emotion(small)
        pose = detect_posture(img)

        self.history.append(conf)
        self.emotions.append(emotion)
        self.posture.append(pose)

        cv2.putText(img, emotion,(20,40),0,1,(0,255,0),2)
        cv2.putText(img, str(round(conf*100,1)),(20,80),0,1,(0,255,0),2)
        cv2.putText(img, pose,(20,120),0,0.7,(255,255,0),2)

        return img


# ================= MAIN =================

ctx = webrtc_streamer(
    key="interview",
    video_transformer_factory=Analyzer
)

# ================= REPORT BUTTON =================

if st.button("📊 Generate Report"):
    if ctx.video_transformer:

        data = ctx.video_transformer

        if len(data.history) > 10:
            st.success("Analysis Complete ✅")

            st.subheader("📈 Confidence Trend")
            st.line_chart(pd.DataFrame({"Confidence": data.history}))

            st.subheader("🎭 Emotion Distribution")
            st.bar_chart(pd.Series(data.emotions).value_counts())

            st.markdown(generate_report(data.history, data.emotions, data.posture))
        else:
            st.warning("Run interview for few seconds first!")
