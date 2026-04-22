# ================================================================
# SCRIPT 01 — Setup & Unzip
# Run this FIRST on Google Colab
# It mounts Drive, installs libraries, and extracts the dataset
# ================================================================

# ── Step 1: Mount Google Drive ───────────────────────────────
# from google.colab import drive
# drive.mount('/content/drive')

# ── Step 2: Install libraries ────────────────────────────────
import subprocess
subprocess.run([
    'pip', 'install',
    'mediapipe==0.10.13',
    'opencv-python-headless',
    'torch', 'torchvision',
    'pandas', 'numpy',
    'matplotlib', 'scikit-learn',
    'tqdm', '-q'
], check=True)
print("Libraries installed!")

# ── Step 3: Download MediaPipe face model ────────────────────
import urllib.request
import os

model_path = '/content/face_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading MediaPipe face landmarker model...")
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/face_landmarker/'
        'face_landmarker/float16/1/face_landmarker.task',
        model_path
    )
    print("Model downloaded!")
else:
    print("Model already exists, skipping download.")

# ── Step 4: Unzip dataset ────────────────────────────────────
import zipfile

# CHANGE THIS to your actual Drive folder path
DRIVE_DATASET_PATH = '/content/drive/MyDrive/DL_project/Drowsiness'
EXTRACT_PATH       = '/content/drive/MyDrive/DL_project/Dataset'  # Colab local SSD — fast!

os.makedirs(EXTRACT_PATH, exist_ok=True)

# We use Fold1 and Fold2 — enough for a strong model
FOLDS_TO_EXTRACT = [
    'Fold1_part1.zip',
    'Fold1_part2.zip',
    'Fold2_part1.zip',
    'Fold2_part2.zip',
]

for fname in FOLDS_TO_EXTRACT:
    fpath = os.path.join(DRIVE_DATASET_PATH, fname)
    if not os.path.exists(fpath):
        print(f"  WARNING: {fname} not found in Drive, skipping.")
        continue
    print(f"Extracting {fname}...")
    with zipfile.ZipFile(fpath, 'r') as z:
        z.extractall(EXTRACT_PATH)
    print(f"  Done: {fname}")

# ── Step 5: Verify structure ─────────────────────────────────
print("\nDataset structure:")
for root, dirs, files in os.walk(EXTRACT_PATH):
    level = root.replace(EXTRACT_PATH, '').count(os.sep)
    indent = '  ' * level
    print(f'{indent}{os.path.basename(root)}/')
    if level == 2:  # show video files
        for f in sorted(files)[:3]:
            print(f'{indent}  {f}')

print("\nSetup complete! Run script 02 next.")
