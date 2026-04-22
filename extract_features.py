# ================================================================
# SCRIPT 02 — Feature Extraction
# Run on Google Colab AFTER script 01
# Extracts eye, mouth, head features from every video
# Saves features_fold1.csv and features_fold2.csv to Drive
# ================================================================

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import (
    FaceLandmarker, FaceLandmarkerOptions, RunningMode
)

# ── Config ────────────────────────────────────────────────────
DATASET_ROOT = '/content/drive/MyDrive/DL_project/Dataset'
DRIVE_OUT    = '/content/drive/MyDrive/DL_project/processed'
MODEL_PATH   = '/content/face_landmarker.task'
os.makedirs(DRIVE_OUT, exist_ok=True)

LABEL_MAP  = {'0': 0, '5': 1, '10': 2}
LABEL_NAME = {0: 'Alert', 1: 'Low-vigilant', 2: 'Drowsy'}

# ── MediaPipe setup ───────────────────────────────────────────
options = FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False
)
detector = FaceLandmarker.create_from_options(options)
print(f"MediaPipe {mp.__version__} ready\n")

# ── Landmark indices ──────────────────────────────────────────
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
MOUTH     = [13,  14,  78,  308, 402, 318, 324, 88]
NOSE_TIP, CHIN, LEFT_EAR, RIGHT_EAR = 1, 152, 234, 454

# ── Feature functions ─────────────────────────────────────────
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
    l_ear = np.array([lm[LEFT_EAR].x * w,  lm[LEFT_EAR].y * h])
    r_ear = np.array([lm[RIGHT_EAR].x * w, lm[RIGHT_EAR].y * h])
    return (np.linalg.norm(chin - nose) /
            (np.linalg.norm(l_ear - r_ear) + 1e-6))

def compute_perclos(ear_series, fps, window_sec=60, threshold=0.2):
    window  = int(fps * window_sec)
    perclos = []
    for i in range(len(ear_series)):
        start = max(0, i - window)
        chunk = ear_series[start:i+1]
        perclos.append(sum(1 for e in chunk if e < threshold) /
                       max(len(chunk), 1))
    return perclos

# ── Process one video ─────────────────────────────────────────
def process_video(video_path, label, subject_id, fold):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'  ERROR: Cannot open {video_path}')
        return None

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rows, frame_idx, no_face = [], 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w   = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_img)

        if result.face_landmarks and len(result.face_landmarks) > 0:
            lm       = result.face_landmarks[0]
            ear      = (eye_aspect_ratio(lm, LEFT_EYE,  w, h) +
                        eye_aspect_ratio(lm, RIGHT_EYE, w, h)) / 2.0
            mar      = mouth_aspect_ratio(lm, w, h)
            tilt     = head_tilt_ratio(lm, w, h)
            face_ok  = 1
        else:
            ear, mar, tilt, face_ok = 0.0, 0.0, 0.0, 0
            no_face += 1

        rows.append({
            'fold'         : fold,
            'subject_id'   : subject_id,
            'label'        : label,
            'frame_idx'    : frame_idx,
            'timestamp_sec': round(frame_idx / fps, 3),
            'ear'          : round(ear,  4),
            'mar'          : round(mar,  4),
            'head_tilt'    : round(tilt, 4),
            'face_detected': face_ok
        })
        frame_idx += 1

        if frame_idx % 1000 == 0:
            pct = (frame_idx / total_frames * 100) if total_frames > 0 else 0
            print(f'    {frame_idx}/{total_frames} ({pct:.0f}%)', end='\r')

    cap.release()
    df             = pd.DataFrame(rows)
    df['perclos']  = compute_perclos(df['ear'].tolist(), fps)
    face_rate      = (frame_idx - no_face) / max(frame_idx, 1) * 100
    print(f'    {frame_idx} frames | face detected: {face_rate:.0f}%')
    return df

# ── Process fold ──────────────────────────────────────────────
def process_fold(fold_name):
    fold_path = os.path.join(DATASET_ROOT, fold_name)
    if not os.path.exists(fold_path):
        print(f"Fold not found: {fold_path}")
        return None

    all_dfs  = []
    subjects = sorted([
        s for s in os.listdir(fold_path)
        if os.path.isdir(os.path.join(fold_path, s))
    ])
    print(f"\nFold: {fold_name} | {len(subjects)} subjects: {subjects}")

    for subject in tqdm(subjects, desc=fold_name):
        subject_path = os.path.join(fold_path, subject)
        for fname in sorted(os.listdir(subject_path)):
            if not fname.lower().endswith(('.mov', '.mp4', '.avi')):
                continue
            label_key = fname.split('.')[0]
            if label_key not in LABEL_MAP:
                continue
            label      = LABEL_MAP[label_key]
            video_path = os.path.join(subject_path, fname)
            print(f'\n  Subject {subject} | {LABEL_NAME[label]} | {fname}')
            df = process_video(video_path, label, subject, fold_name)
            if df is not None and len(df) > 0:
                all_dfs.append(df)

    if not all_dfs:
        return None

    fold_df  = pd.concat(all_dfs, ignore_index=True)
    out_path = os.path.join(DRIVE_OUT, f'features_{fold_name}.csv')
    fold_df.to_csv(out_path, index=False)
    print(f'\nSaved {len(fold_df):,} rows → {out_path}')
    print('Class distribution:')
    for lbl, cnt in fold_df.groupby('label').size().items():
        print(f'  {LABEL_NAME[lbl]:15s}: {cnt:,} frames')
    return fold_df

# ── Smart resume — skips already processed folds ──────────────
all_folds = sorted([
    f for f in os.listdir(DATASET_ROOT)
    if os.path.isdir(os.path.join(DATASET_ROOT, f))
])
print(f"All folds found: {all_folds}\n")

all_results = []
for fold in all_folds:
    out_path = os.path.join(DRIVE_OUT, f'features_{fold}.csv')

    if os.path.exists(out_path):
        # Already processed — load from Drive, skip extraction
        print(f"SKIPPING {fold} — CSV already exists in Drive")
        all_results.append(pd.read_csv(out_path))
        continue

    # Not processed yet — run extraction
    result = process_fold(fold)
    if result is not None:
        all_results.append(result)

# ── Combine all into master CSV ───────────────────────────────
if all_results:
    master_df  = pd.concat(all_results, ignore_index=True)
    master_out = os.path.join(DRIVE_OUT, 'features_all.csv')
    master_df.to_csv(master_out, index=False)
    print(f'\n{"="*55}')
    print(f'MASTER CSV saved: {len(master_df):,} rows → {master_out}')
    print(f'Subjects  : {master_df["subject_id"].nunique()}')
    print(f'Folds     : {master_df["fold"].nunique()}')
    print('\nOverall class distribution:')
    for lbl, cnt in master_df.groupby('label').size().items():
        print(f'  {LABEL_NAME[lbl]:15s}: {cnt:,} frames')
    print('\nDone! Run 03_eda_visualise.py next.')
# ```

# ---

# **Key changes from the old script:**

# `DATASET_ROOT` now points directly to `/content/drive/MyDrive/DL_project/Dataset` — no unzipping needed ever again. The smart resume checks `/content/drive/MyDrive/DL_project/processed/` for existing CSVs. `Fold1_part1` CSV is already there so it will print `SKIPPING Fold1_part1` and jump straight to `Fold1_part2`.

# ---

# **What you'll see when you run it:**
# ```
# All folds found: ['Fold1_part1', 'Fold1_part2', 'Fold2_part1', 'Fold2_part2']

# SKIPPING Fold1_part1 — CSV already exists in Drive

# Fold: Fold1_part1 | 6 subjects: ...   ← loads in seconds

# Fold: Fold1_part2 | 6 subjects: ...   ← starts processing here
#   Subject 07 | Alert | 0.mov
#     500/18000 (3%)...