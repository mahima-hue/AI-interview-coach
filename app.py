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
HEADERS = {"Authorization": "Bearer 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"}  # <-- put your token here

# ================= SESSION STORAGE =================
if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.emotions = []
    st.session_state.postures = []

# ================= UI =================
st.set_page_config(page_title="AI Interview Coach", layout="wide")

st.title("🎤 AI Interview Behavior Analyzer")
st.markdown("Real-time AI Mock Interview System")

# RESET BUTTON
if st.button("🔄 Reset Session"):
    st.session_state.history = []
    st.session_state.emotions = []
    st.session_state.postures = []
    st.success("Session Reset!")

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


# ================= REPORT =================

def generate_report(hist, emo, post):

    hist = np.array(hist)
    avg = np.mean(hist)
    std = np.std(hist)

    trend = "improving" if hist[-1] > hist[0] else "declining" if hist[-1] < hist[0] else "stable"

    dominant = Counter(emo).most_common(1)[0][0]
    posture_main = Counter(post).most_common(1)[0][0]

    return f"""
## 🧠 AI Interview Report

### 📊 Confidence
Average: **{round(avg*10)}/10**  
Trend: **{trend}**  
Stability: **{'high' if std<0.1 else 'medium' if std<0.2 else 'low'}**

### 🎭 Emotion
Dominant: **{dominant}**

### 🧍 Posture
Mostly: **{posture_main}**

---

### 🚀 Recommendations
• Maintain consistent confidence  
• Improve facial expressions  
• Keep eye contact  
• Practice regularly  

---

### 🏁 Verdict
{'Strong performance' if avg>0.7 else 'Needs improvement'}
"""


# ================= VIDEO CLASS =================

class Analyzer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        small = cv2.resize(img,(224,224))

        emotion, conf = get_emotion(small)
        pose = detect_posture(img)

        # STORE DATA (PERSISTENT)
        st.session_state.history.append(conf)
        st.session_state.emotions.append(emotion)
        st.session_state.postures.append(pose)

        # DISPLAY TEXT
        cv2.putText(img, emotion,(20,40),0,1,(0,255,0),2)
        cv2.putText(img, str(round(conf*100,1)),(20,80),0,1,(0,255,0),2)
        cv2.putText(img, pose,(20,120),0,0.7,(255,255,0),2)

        return img


# ================= CAMERA =================

webrtc_streamer(
    key="interview",
    video_transformer_factory=Analyzer
)

# ================= REPORT BUTTON =================

st.markdown("---")

if st.button("📊 Generate Report"):

    hist = st.session_state.history
    emo = st.session_state.emotions
    post = st.session_state.postures

    if len(hist) > 5:

        st.success("Analysis Complete ✅")

        st.subheader("📈 Confidence Trend")
        st.line_chart(pd.DataFrame({"Confidence": hist}))

        st.subheader("🎭 Emotion Distribution")
        st.bar_chart(pd.Series(emo).value_counts())

        st.markdown(generate_report(hist, emo, post))

    else:
        st.warning("⚠️ Run interview for at least 5–10 seconds first!")
