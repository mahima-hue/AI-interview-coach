import streamlit as st
import cv2
import requests
import base64
import time
import numpy as np
import pandas as pd
from collections import Counter

# ================= CONFIG =================
API_URL = "https://router.huggingface.co/hf-inference/models/trpakov/vit-face-expression"
HEADERS = {"Authorization": "Bearer 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"}

# ================= UI =================
st.set_page_config(page_title="AI Interview Coach", layout="wide")

st.markdown("""
<style>
body {background-color:#0e1117; color:white;}
h1, h2, h3 {text-align:center;}
.block-container {padding-top:1rem;}
</style>
""", unsafe_allow_html=True)

st.title("🎤 AI Interview Behavior Analyzer")
st.markdown("<p style='text-align:center;'>Real-time Behavioral Intelligence System</p>", unsafe_allow_html=True)

col_cam, col_panel = st.columns([2,1])
tip_box = col_panel.empty()
progress_chart = col_panel.empty()
metric_box = col_panel.empty()

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


def get_live_tip(conf, emotion):
    if conf < 0.5:
        return "⚠️ Confidence dropping — slow down and speak clearly"
    elif emotion in ["sad","fear"]:
        return "⚠️ You seem tense — relax your face"
    elif emotion == "neutral":
        return "💡 Add more expressions"
    elif emotion == "happy":
        return "✅ Great energy — maintain it"
    return "💡 Stay composed"


# ================= 🧠 SMART ANALYSIS =================

def generate_report(history, emotions, posture_history):

    history = np.array(history)
    avg = np.mean(history)
    std = np.std(history)

    # trend
    trend_value = history[-1] - history[0]
    if trend_value > 0.1:
        trend = "improving"
    elif trend_value < -0.1:
        trend = "declining"
    else:
        trend = "stable"

    # emotion distribution
    counts = Counter(emotions)
    total = len(emotions)
    emotion_percent = {k: round(v/total*100,1) for k,v in counts.items()}
    dominant = max(emotion_percent, key=emotion_percent.get)

    # posture consistency
    posture_counts = Counter(posture_history)
    posture_main = max(posture_counts, key=posture_counts.get)
    posture_ratio = posture_counts[posture_main] / len(posture_history)

    # dynamic confidence description
    if avg > 0.75:
        conf_desc = "consistently strong"
    elif avg > 0.6:
        conf_desc = "moderately strong but not fully stable"
    else:
        conf_desc = "inconsistent and needs improvement"

    # stability
    if std < 0.08:
        stability = "highly stable"
    elif std < 0.18:
        stability = "moderately stable"
    else:
        stability = "unstable"

    # emotion interpretation
    if dominant == "happy":
        emotion_desc = "engaging and positive"
    elif dominant == "neutral":
        emotion_desc = "controlled but less expressive"
    else:
        emotion_desc = "low energy or slightly stressed"

    # posture interpretation
    if posture_ratio > 0.75:
        posture_desc = "well aligned and focused"
    else:
        posture_desc = "inconsistent and distracting"

    # FINAL DYNAMIC REPORT
    return f"""
## 🧠 Personalized AI Interview Analysis

### 📊 Confidence Behavior
Your confidence remained **{conf_desc}**, with a **{trend} trajectory** throughout the session.  
The variation suggests your delivery was **{stability}**, indicating how well you handled pressure over time.

---

### 🎭 Emotional Intelligence
You showed a dominant expression of **{dominant} ({emotion_percent[dominant]}%)**, reflecting a **{emotion_desc} communication style**.

---

### 🧍 Posture & Presence
Your posture was **{posture_desc}**, with **{round(posture_ratio*100)}% alignment consistency**, which directly impacts interviewer perception.

---

### ⚡ Behavioral Insight
- You {'adapted well during the session' if trend=='improving' else 'lost confidence gradually' if trend=='declining' else 'maintained a steady pattern'}
- Your engagement level was {'high' if dominant=='happy' else 'moderate' if dominant=='neutral' else 'low'}
- Your body language was {'professional' if posture_ratio>0.7 else 'needs improvement'}

---

## 🚀 Smart Recommendations

• Maintain consistent confidence from start to end  
• Improve facial expressiveness to increase engagement  
• Keep eye contact steady (stay centered)  
• Practice controlled breathing to reduce fluctuations  
• Simulate real interviews regularly to build stability  

---

## 🏁 Final Verdict

You {'demonstrate strong interview readiness' if avg>0.7 else 'show potential but need refinement in delivery and consistency'}.
"""


# ================= MAIN =================

duration = st.slider("⏱ Analysis Duration (seconds)", 5, 120, 30)

if st.button("🚀 Start Analysis"):

    cap = cv2.VideoCapture(0)

    history = []
    emotions = []
    posture_history = []

    frame_box = col_cam.empty()

    start = time.time()
    last_api = 0
    frame_count = 0

    current_emotion = "neutral"
    current_conf = 0.5

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame,1)
        small = cv2.resize(frame,(224,224))

        frame_count += 1

        if frame_count % 5 == 0 and time.time() - last_api > 2:
            current_emotion, current_conf = get_emotion(small)
            last_api = time.time()

        posture = detect_posture(frame)

        history.append(current_conf)
        emotions.append(current_emotion)
        posture_history.append(posture)

        # LIVE UI
        df_live = pd.DataFrame({"Confidence": history})
        progress_chart.line_chart(df_live)

        metric_box.metric("Confidence", round(current_conf*100,1))
        tip_box.markdown(f"### 💡 {get_live_tip(current_conf, current_emotion)}")

        # overlay
        cv2.putText(frame, current_emotion,(20,40),0,1,(0,255,0),2)
        cv2.putText(frame, str(round(current_conf*100,1)), (20,80),0,1,(0,255,0),2)
        cv2.putText(frame, posture,(20,120),0,0.7,(255,255,0),2)

        frame_box.image(frame, channels="BGR")

        time.sleep(0.03)

        if time.time() - start > duration:
            break

    cap.release()

    st.success("✅ Analysis Completed")

    # DASHBOARD
    st.subheader("🏆 Performance Breakdown")

    conf = round(np.mean(history)*10)
    stab = round((1-np.std(history))*10)
    expr = round((emotions.count("happy")/len(emotions))*10)
    post = round((posture_history.count("Centered")/len(posture_history))*10)

    col1,col2,col3,col4 = st.columns(4)
    col1.metric("Confidence", conf)
    col2.metric("Stability", stab)
    col3.metric("Expression", expr)
    col4.metric("Posture", post)

    st.subheader("📊 Detailed Scores")
    st.progress(conf/10)
    st.progress(stab/10)
    st.progress(expr/10)
    st.progress(post/10)

    st.subheader("🎭 Emotion Distribution")
    st.bar_chart(pd.Series(emotions).value_counts())

    st.subheader("📈 Confidence Trend")
    st.line_chart(pd.DataFrame({"Confidence": history}))

    st.markdown(generate_report(history, emotions, posture_history))