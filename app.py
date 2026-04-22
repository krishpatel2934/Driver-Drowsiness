# ================================================================
# app.py — Real-Time Drowsiness Detection (Streamlit)
# Model: BiGRU + Multi-Head Attention + Focal Loss (v5)
#
# Fixes vs original:
#   1. PERCLOS computed as rolling 60-sec window % (not binary per frame)
#   2. Inference seq_len = 150 frames (5 sec) not 600 — first score in 5s
#   3. Display skipped on non-inference frames — much faster
#   4. Baseline collection extended to 450 frames (15 seconds)
#   5. Score smoothed over last 5 predictions — stable gauge
# ================================================================

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
import cv2
import numpy as np
import torch
import pickle
import time
import mediapipe as mp
from collections import deque
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)
from models import BiGRU_MHA
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



st.set_page_config(
    page_title="Drowsiness Detection — BiGRU v5",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Constants ─────────────────────────────────────────────────
# FIX 2: use 150 frames (5 sec) for first score, not 600
INFER_SEQ_LEN   = 150
TRAIN_SEQ_LEN   = 600
N_FEATURES      = 16

# FIX 4: 450 frames (~15 sec) for reliable baseline
BASELINE_FRAMES = 450

EAR_CLOSE_THR   = 0.21

# FIX 1: rolling PERCLOS window — 60 seconds at 30fps
PERCLOS_WINDOW  = 1800

ROLL_WINDOW     = 90

MODEL_PATH      = "v5_focal_models/drowsiness_bigru_mha_v5_focal.pth"
SCALER_PATH     = "v5_focal_models/scaler_v5.pkl"
LANDMARKER_PATH = "face_landmarker.task"

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
MOUTH     = [13,  14,  78,  308, 402, 318, 324, 88]
NOSE_TIP  = 1
CHIN      = 152
L_EAR_IDX = 234
R_EAR_IDX = 454

LEVEL_CFG = {
    "safe":    {"label": "✅  ALERT",   "cls": "badge-safe",    "color": "#4ade80"},
    "warning": {"label": "⚠️  CAUTION", "cls": "badge-warning", "color": "#fbbf24"},
    "danger":  {"label": "🚨  DROWSY",  "cls": "badge-danger",  "color": "#f87171"},
}

# ================================================================
# CSS
# ================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
html, body { overflow-x: hidden !important; }
html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #111827 50%, #0f172a 100%);
    overflow-x: hidden !important;
}
.hero-title {
    text-align: left; font-size: 1.8rem; font-weight: 800;
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.1rem; letter-spacing: -0.03em;
}
.hero-sub { text-align: left; color: #94a3b8; font-size: 0.8rem; margin-bottom: 0; }
.badge-wrap { text-align: center; margin-bottom: 0.5rem; }
.alert-badge {
    display: inline-block; padding: 0.45rem 1.4rem;
    border-radius: 999px; font-weight: 700; font-size: 1.05rem; letter-spacing: 0.06em;
}
.badge-safe    { background: rgba(34,197,94,0.15);  color: #4ade80; border: 1.5px solid rgba(34,197,94,0.35); }
.badge-warning { background: rgba(250,204,21,0.15); color: #fbbf24; border: 1.5px solid rgba(250,204,21,0.35); }
.badge-danger  {
    background: rgba(239,68,68,0.15); color: #f87171; border: 1.5px solid rgba(239,68,68,0.35);
    box-shadow: 0 0 18px rgba(239,68,68,0.4); animation: dpulse 1s ease-in-out infinite;
}
@keyframes dpulse {
    0%,100% { box-shadow: 0 0 10px rgba(239,68,68,0.3); }
    50%      { box-shadow: 0 0 26px rgba(239,68,68,0.7); }
}
.gauge-wrap  { text-align: center; margin: 0.1rem 0 0.2rem; }
.gauge-score { font-size: 40px; font-weight: 800; font-family: Inter, sans-serif; line-height: 1.15; }
.gauge-label { font-size: 11px; color: #94a3b8; font-family: Inter, sans-serif; margin-top: 1px; }
.pbar-wrap  { margin-bottom: 0.5rem; }
.pbar-label { font-size: 0.72rem; font-weight: 600; color: #cbd5e1; margin-bottom: 0.2rem; }
.pbar-track { background: rgba(51,65,85,0.5); border-radius: 999px; height: 9px; overflow: hidden; }
.pbar-fill  { height: 100%; border-radius: 999px; transition: width 0.25s ease; }
.feat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.45rem; margin-top: 0.5rem; }
.feat-pill {
    background: rgba(30,41,59,0.6); border: 1px solid rgba(148,163,184,0.1);
    border-radius: 11px; padding: 0.5rem 0.75rem;
    display: flex; justify-content: space-between; align-items: center;
}
.feat-name { font-size: 0.65rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em; }
.feat-val  { font-size: 0.95rem; font-weight: 700; color: #e2e8f0; }
.calib-text { text-align: center; color: #94a3b8; font-size: 0.75rem; margin-top: 0.3rem; }
.meta-bar {
    text-align: center; color: #475569; font-size: 0.68rem;
    padding: 0.4rem 0; border-top: 1px solid rgba(148,163,184,0.08); margin-top: 0.5rem;
}
.vid-placeholder {
    display: flex; align-items: center; justify-content: center;
    height: 360px; background: rgba(30,41,59,0.4);
    border-radius: 14px; border: 1px solid rgba(148,163,184,0.1);
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0.5rem !important; padding-bottom: 0 !important; }
img { max-width: 100% !important; height: auto !important; }
</style>
""", unsafe_allow_html=True)


# ================================================================
# Feature helpers
# ================================================================
def eye_aspect_ratio(lm, indices, w, h):
    pts = [(lm[i].x * w, lm[i].y * h) for i in indices]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C + 1e-6)

def mouth_aspect_ratio(lm, w, h):
    pts   = [(lm[i].x * w, lm[i].y * h) for i in MOUTH]
    vert  = np.linalg.norm(np.array(pts[0]) - np.array(pts[1]))
    horiz = np.linalg.norm(np.array(pts[2]) - np.array(pts[3]))
    return vert / (horiz + 1e-6)

def head_tilt_ratio(lm, w, h):
    nose  = np.array([lm[NOSE_TIP].x * w, lm[NOSE_TIP].y * h])
    chin  = np.array([lm[CHIN].x * w,     lm[CHIN].y * h])
    l_ear = np.array([lm[L_EAR_IDX].x * w, lm[L_EAR_IDX].y * h])
    r_ear = np.array([lm[R_EAR_IDX].x * w, lm[R_EAR_IDX].y * h])
    return np.linalg.norm(chin - nose) / (np.linalg.norm(l_ear - r_ear) + 1e-6)


# ================================================================
# Load model / scaler / detector (cached — only once)
# ================================================================
@st.cache_resource
def load_assets():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = BiGRU_MHA(
        input_size=16, hidden_size=128,
        num_layers=2, num_classes=3,
        dropout=0.3, num_heads=4,
    ).to(device)
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    options = FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=LANDMARKER_PATH),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
    )
    detector = FaceLandmarker.create_from_options(options)
    return model, scaler, detector, device, ckpt.get("loso_f1", None)


# ================================================================
# Session state
# ================================================================
def init_state():
    s = st.session_state
    s["main_buf"]       = deque(maxlen=INFER_SEQ_LEN)  # FIX 2: 150 not 600
    s["roll_ear"]       = deque(maxlen=ROLL_WINDOW)
    s["roll_mar"]       = deque(maxlen=ROLL_WINDOW)
    s["roll_tilt"]      = deque(maxlen=ROLL_WINDOW)
    s["roll_perclos"]   = deque(maxlen=ROLL_WINDOW)
    # FIX 1: dedicated PERCLOS window (60 sec at 30fps)
    s["ear_perclos_buf"] = deque(maxlen=PERCLOS_WINDOW)
    s["baseline_list"]  = []
    s["baseline_ready"] = False
    s["baseline_mean"]  = np.zeros(4, dtype=np.float32)
    s["baseline_std"]   = np.ones(4,  dtype=np.float32)
    s["prev_base"]      = None
    s["frame_idx"]      = 0
    s["score"]          = 0.0
    s["probs"]          = np.array([1.0, 0.0, 0.0])
    s["level"]          = "safe"
    s["score_history"]  = deque(maxlen=200)
    # FIX 5: score smoothing buffer
    s["score_smooth"]   = deque(maxlen=5)
    s["running"]        = False

if "main_buf" not in st.session_state:
    init_state()


# ================================================================
# HTML widget builders
# ================================================================
def gauge_html(score: float) -> str:
    pct    = min(max(float(score), 0.0), 100.0) / 100.0
    colour = "#f87171" if pct >= 0.65 else "#fbbf24" if pct >= 0.35 else "#4ade80"
    cx, cy, r = 130, 130, 100
    sx    = cx - r
    ex_bg = cx + r
    angle_rad = np.radians(180.0 - pct * 180.0)
    ex = cx + r * np.cos(angle_rad)
    ey = cy - r * np.sin(angle_rad)
    bg_path = f"M {sx},{cy} A {r},{r} 0 1,1 {ex_bg},{cy}"
    fg_tag  = "" if pct < 0.001 else (
        f'<path d="M {sx},{cy} A {r},{r} 0 0,1 {ex:.2f},{ey:.2f}" '
        f'fill="none" stroke="{colour}" stroke-width="13" stroke-linecap="round"/>'
    )
    return (
        f'<div class="gauge-wrap">'
        f'<svg viewBox="0 22 260 116" style="width:210px;height:94px;display:block;margin:0 auto;">'
        f'<path d="{bg_path}" fill="none" stroke="rgba(51,65,85,0.45)" stroke-width="13" stroke-linecap="round"/>'
        f'{fg_tag}</svg>'
        f'<div class="gauge-score" style="color:{colour};">{score:.0f}</div>'
        f'<div class="gauge-label">Fatigue Score</div>'
        f'</div>'
    )


def prob_bars_html(probs: np.ndarray) -> str:
    names  = ["Alert", "Low-vigilant", "Drowsy"]
    colors = ["#4ade80", "#fbbf24", "#f87171"]
    html   = ""
    for n, c, p in zip(names, colors, probs):
        w = max(float(p) * 100, 0.4)
        html += (
            f'<div class="pbar-wrap">'
            f'<div class="pbar-label">{n} — {p*100:.1f}%</div>'
            f'<div class="pbar-track">'
            f'<div class="pbar-fill" style="width:{w:.1f}%;background:{c};"></div>'
            f'</div></div>'
        )
    return html


def feat_pills_html(feats: dict) -> str:
    items = "".join(
        f'<div class="feat-pill">'
        f'<span class="feat-name">{name}</span>'
        f'<span class="feat-val">{val:.3f}</span>'
        f'</div>'
        for name, val in feats.items()
    )
    return f'<div class="feat-grid">{items}</div>'


# ================================================================
# Feature engineering — one frame → 16-dim vector
# FIX 1: PERCLOS now computed as rolling window %, matching training
# ================================================================
def process_one_frame(lm, w, h) -> dict:
    s = st.session_state

    ear  = (eye_aspect_ratio(lm, LEFT_EYE, w, h) +
            eye_aspect_ratio(lm, RIGHT_EYE, w, h)) / 2.0
    mar  = mouth_aspect_ratio(lm, w, h)
    tilt = head_tilt_ratio(lm, w, h)

    # FIX 1: proper rolling PERCLOS — same as training
    s["ear_perclos_buf"].append(ear)
    buf     = s["ear_perclos_buf"]
    perclos = sum(1 for e in buf if e < EAR_CLOSE_THR) / max(len(buf), 1)

    base = np.array([ear, mar, tilt, perclos], dtype=np.float32)

    # Baseline calibration — FIX 4: 450 frames for reliability
    if not s["baseline_ready"]:
        s["baseline_list"].append(base.copy())
        if len(s["baseline_list"]) >= BASELINE_FRAMES:
            bl = np.array(s["baseline_list"], dtype=np.float32)
            s["baseline_mean"] = bl.mean(axis=0)
            s["baseline_std"]  = bl.std(axis=0) + 1e-6
            s["baseline_ready"] = True
        pn = np.zeros(4, dtype=np.float32)
    else:
        pn = (base - s["baseline_mean"]) / s["baseline_std"]

    delta = (base - s["prev_base"]
             if s["prev_base"] is not None
             else np.zeros(4, dtype=np.float32))
    s["prev_base"] = base.copy()

    s["roll_ear"].append(ear);   s["roll_mar"].append(mar)
    s["roll_tilt"].append(tilt); s["roll_perclos"].append(perclos)
    roll = np.array([np.mean(s["roll_ear"]),    np.mean(s["roll_mar"]),
                     np.mean(s["roll_tilt"]),   np.mean(s["roll_perclos"])],
                    dtype=np.float32)

    s["main_buf"].append(np.concatenate([base, pn, delta, roll]))
    s["frame_idx"] += 1
    return {"EAR": ear, "MAR": mar, "Head Tilt": tilt, "PERCLOS": perclos}


# ================================================================
# Inference
# FIX 2: uses INFER_SEQ_LEN (150) so first score fires after 5 sec
# FIX 5: score smoothed over last 5 predictions
# ================================================================
def run_inference(model, scaler, device):
    s = st.session_state
    if len(s["main_buf"]) < INFER_SEQ_LEN:
        return
    arr    = np.array(s["main_buf"], dtype=np.float32)
    tensor = torch.tensor(
        scaler.transform(arr)[None], dtype=torch.float32
    ).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=-1).cpu().numpy()[0]

    raw_score = float(probs[1] * 50 + probs[2] * 100)

    # FIX 5: smooth score
    s["score_smooth"].append(raw_score)
    score     = float(np.mean(s["score_smooth"]))

    s["score"] = score
    s["probs"] = probs
    s["level"] = "danger" if score >= 65 else "warning" if score >= 35 else "safe"
    s["score_history"].append(score)


# ================================================================
# Draw overlays on frame
# ================================================================
def draw_overlays(frame, lm, w, h, face_ok):
    if face_ok and lm is not None:
        # Eye landmarks
        for indices, col in [(LEFT_EYE, (96,165,250)), (RIGHT_EYE, (96,165,250))]:
            pts = [(int(lm[i].x*w), int(lm[i].y*h)) for i in indices]
            for p in pts:
                cv2.circle(frame, p, 2, col, -1)
            for i in range(len(pts)):
                cv2.line(frame, pts[i], pts[(i+1) % len(pts)], col, 1)
        # Mouth landmarks
        for p in [(int(lm[i].x*w), int(lm[i].y*h)) for i in MOUTH]:
            cv2.circle(frame, p, 2, (167,139,250), -1)
        # Nose-chin line (head tilt indicator)
        cv2.line(frame,
                 (int(lm[NOSE_TIP].x*w), int(lm[NOSE_TIP].y*h)),
                 (int(lm[CHIN].x*w),     int(lm[CHIN].y*h)),
                 (244,114,182), 1)

    # Status dot top-left
    dot_c = {"safe":(74,222,128), "warning":(251,191,36), "danger":(248,113,113)}
    cv2.circle(frame, (18,18), 8,
               dot_c.get(st.session_state.get("level","safe"), (74,222,128)), -1)

    if not face_ok:
        cv2.putText(frame, "No face detected", (34,23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100,100,255), 1)
    return frame


# ================================================================
# Load assets
# ================================================================
model, scaler, detector, device, loso_f1 = load_assets()

# ================================================================
# Header + Controls
# ================================================================
head_left, head_right = st.columns([5, 1.5])
with head_left:
    st.markdown('<div class="hero-title">🧠 Drowsiness Detection</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">BiGRU + Multi-Head Attention · Focal Loss v5 · Real-Time</div>',
                unsafe_allow_html=True)
with head_right:
    c1, c2 = st.columns(2)
    with c1:
        start = st.button("▶ Start", type="primary", use_container_width=True)
    with c2:
        stop  = st.button("■ Stop",  use_container_width=True)

if start:
    init_state()
    st.session_state["running"] = True
if stop:
    st.session_state["running"] = False

# ================================================================
# Layout
# ================================================================
col_vid, col_dash = st.columns([5, 3], gap="medium")

with col_vid:
    video_slot = st.empty()
    buf_bar    = st.empty()

with col_dash:
    badge_slot = st.empty()
    gauge_slot = st.empty()
    probs_slot = st.empty()
    feats_slot = st.empty()
    chart_slot = st.empty()
    meta_slot  = st.empty()


# ================================================================
# Dashboard renderer
# ================================================================
def render_dashboard(feats=None):
    s   = st.session_state
    lvl = s["level"]
    cfg = LEVEL_CFG[lvl]

    badge_slot.markdown(
        f'<div class="badge-wrap">'
        f'<span class="alert-badge {cfg["cls"]}">{cfg["label"]}</span>'
        f'</div>', unsafe_allow_html=True)

    gauge_slot.markdown(gauge_html(s["score"]), unsafe_allow_html=True)
    probs_slot.markdown(prob_bars_html(s["probs"]), unsafe_allow_html=True)

    if s["running"]:
        if feats:
            feats_slot.markdown(feat_pills_html(feats), unsafe_allow_html=True)
        hist = list(s["score_history"])
        if len(hist) > 1:
            chart_slot.line_chart(hist, height=90, use_container_width=True)
        else:
            chart_slot.empty()
    else:
        feats_slot.empty()
        chart_slot.empty()

    buf_len = len(s["main_buf"])
    f1_str  = f" · LOSO F1 {loso_f1:.3f}" if loso_f1 else ""
    calib   = "✓ Calibrated" if s.get("baseline_ready") else \
              f"Calibrating… {len(s['baseline_list'])}/{BASELINE_FRAMES}"

    meta_slot.markdown(
        f'<div class="meta-bar">'
        f'Buf {buf_len}/{INFER_SEQ_LEN} · Frame #{s["frame_idx"]} · '
        f'{device.upper()} · {calib}{f1_str}'
        f'</div>', unsafe_allow_html=True)


# ================================================================
# Camera loop
# FIX 3: display and inference run at different rates
#   inference  → every 3 frames  (~10fps inference)
#   display    → every 2 frames  (~15fps video)
#   dashboard  → every 3 frames  (in sync with inference)
# ================================================================
if st.session_state["running"]:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    # Reduce internal buffer so frames are fresh not stale
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        st.error("❌ Cannot open webcam — check permissions.")
        st.session_state["running"] = False
    else:
        frame_cnt   = 0
        last_feats  = {}

        while st.session_state["running"]:
            ret, frame = cap.read()
            if not ret:
                continue

            frame  = cv2.flip(frame, 1)
            h, w   = frame.shape[:2]
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

            face_ok = bool(result.face_landmarks)
            lm      = result.face_landmarks[0] if face_ok else None

            if face_ok:
                last_feats = process_one_frame(lm, w, h)
            else:
                st.session_state["main_buf"].append(
                    np.zeros(N_FEATURES, dtype=np.float32))
                st.session_state["frame_idx"] += 1

            frame_cnt += 1

            # FIX 3: inference + dashboard every 3 frames
            if frame_cnt % 3 == 0:
                run_inference(model, scaler, device)
                render_dashboard(last_feats)

            # FIX 3: display every 2 frames for smoother video
            if frame_cnt % 2 == 0:
                video_slot.image(
                    draw_overlays(rgb.copy(), lm, w, h, face_ok),
                    channels="RGB", width=480)

            # Progress bar until buffer full
            buf_len = len(st.session_state["main_buf"])
            if buf_len < INFER_SEQ_LEN:
                buf_bar.progress(
                    buf_len / INFER_SEQ_LEN,
                    text=f"Warming up… {buf_len}/{INFER_SEQ_LEN} frames (5 sec)")
            else:
                buf_bar.empty()

            if not st.session_state.get("running", False):
                break

        cap.release()

else:
    video_slot.markdown(
        '<div class="vid-placeholder">'
        '<div style="text-align:center;">'
        '<div style="font-size:3rem;margin-bottom:0.5rem;">📷</div>'
        '<div style="color:#94a3b8;font-size:0.9rem;">Press <b>▶ Start</b> to begin detection</div>'
        '</div></div>', unsafe_allow_html=True)
    render_dashboard()