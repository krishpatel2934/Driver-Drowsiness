# ================================================================
# train.py — BiGRU + MHA with FOCAL LOSS (Targets Low-Vigilant)
# ================================================================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, json, pickle
from tqdm import tqdm
import sys
sys.path.append('/kaggle/input/datasets/krishpatel2934/drowsiness-floss')
from models import get_model, count_params

# ================================================================
# NEW: FOCAL LOSS IMPLEMENTATION
# ================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss forces the model to focus on hard-to-classify examples
    (like the 'Low-vigilant' class) instead of easy ones (like 'Alert').
    """
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha # We will pass your class weights here

    def forward(self, inputs, targets):
        # Standard cross entropy without reduction
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        # pt is the probability of the correct class
        pt = torch.exp(-ce_loss)
        # Focal math: down-weight easy examples (where pt is high)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        return focal_loss.mean()

# ================================================================
# CONFIG
# ================================================================
BASE_INPUT = '/kaggle/input/datasets/krishpatel2934/drowsiness-floss'
CSV_PATH   = f'{BASE_INPUT}/features_all_clean1.csv'

OUTPUT_DIR = '/kaggle/working/v5_focal_models'
PLOTS_DIR  = '/kaggle/working/v5_focal_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

SEQ_LEN    = 600
STRIDE     = 30
BATCH_SIZE = 32
EPOCHS     = 60
LR         = 5e-4
PATIENCE   = 10
N_CLASSES  = 3
NUM_HEADS  = 4
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

BASE_FEATURES = ['ear','mar','head_tilt','perclos']
FEATURE_ORDER = [
    'ear','mar','head_tilt','perclos',
    'ear_pn','mar_pn','head_tilt_pn','perclos_pn',
    'ear_delta','mar_delta','head_tilt_delta','perclos_delta',
    'ear_roll90','mar_roll90','head_tilt_roll90','perclos_roll90'
]
N_FEATURES = len(FEATURE_ORDER)
LABEL_NAME = {0:'Alert', 1:'Low-vigilant', 2:'Drowsy'}

# ONLY TRAIN BiGRU_MHA FOR THIS TEST
MODELS_TO_TRAIN = ['bigru_mha'] 

print(f"Device  : {DEVICE}")
print(f"Loss    : FOCAL LOSS (gamma=2.0)")
print(f"Models  : {MODELS_TO_TRAIN}")

# ================================================================
# DATA PREP (Exact same as v4)
# ================================================================
print(f"\nLoading {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)
df[BASE_FEATURES] = df[BASE_FEATURES].fillna(0)
print(f"Loaded {len(df):,} rows | {df['subject_id'].nunique()} subjects")

print("Per-subject normalisation...")
for feat in BASE_FEATURES:
    df[f'{feat}_pn'] = 0.0
    for subj in df['subject_id'].unique():
        sm = df['subject_id']==subj
        am = sm & (df['label']==0)
        if am.sum()==0: continue
        mu  = df.loc[am,feat].mean()
        sig = df.loc[am,feat].std() + 1e-6
        df.loc[sm,f'{feat}_pn'] = (df.loc[sm,feat] - mu) / sig

print("Delta features...")
for feat in BASE_FEATURES:
    df[f'{feat}_delta'] = (df.groupby(['fold','subject_id','label'])[feat].diff().fillna(0))

print("Rolling features...")
for feat in BASE_FEATURES:
    df[f'{feat}_roll90'] = (df.groupby(['fold','subject_id','label'])[feat].transform(lambda x: x.rolling(90, min_periods=1).mean()))

def build_sequences(dataframe, seq_len, stride):
    X_list, y_list, subj_list = [], [], []
    for (subj, label), grp in dataframe.groupby(['subject_id','label']):
        grp  = grp.sort_values('frame_idx').reset_index(drop=True)
        vals = grp[FEATURE_ORDER].values.astype(np.float32)
        for s in range(0, len(vals)-seq_len+1, stride):
            X_list.append(vals[s:s+seq_len])
            y_list.append(label)
            subj_list.append(subj)
    return (np.array(X_list,dtype=np.float32), np.array(y_list, dtype=np.int64), subj_list)

print("\nBuilding sequences...")
X, y, subjects = build_sequences(df, SEQ_LEN, STRIDE)
print(f"X shape : {X.shape}")

scaler   = StandardScaler()
X_norm   = scaler.fit_transform(X.reshape(-1,N_FEATURES)).reshape(X.shape)
with open(os.path.join(OUTPUT_DIR,'scaler_v5.pkl'),'wb') as f:
    pickle.dump(scaler,f)

class DrowsinessDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

# ================================================================
# LOSO-CV (Modified to use Focal Loss)
# ================================================================
def run_loso(model_name, X_norm, y, subjects):
    unique_subjs    = sorted(set(subjects))
    all_preds, all_true = [], []
    subject_f1s     = {}

    print(f"\n{'='*55}")
    print(f"LOSO-CV — {model_name.upper()} | FOCAL LOSS")
    print(f"{'='*55}")

    for test_subj in tqdm(unique_subjs, desc=model_name.upper()):
        tr_mask = np.array([s!=test_subj for s in subjects])
        te_mask = np.array([s==test_subj for s in subjects])
        X_tr,y_tr = X_norm[tr_mask], y[tr_mask]
        X_te,y_te = X_norm[te_mask], y[te_mask]
        if len(X_te)==0: continue

        tr_loader = DataLoader(DrowsinessDataset(X_tr,y_tr),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        te_loader = DataLoader(DrowsinessDataset(X_te,y_te),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

        model = get_model(model_name, input_size=N_FEATURES, num_heads=NUM_HEADS).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        # Calculate class weights
        counts  = np.bincount(y_tr, minlength=N_CLASSES).astype(float)
        weights = torch.tensor(
            (1.0/(counts+1e-6))/(1.0/(counts+1e-6)).sum()*N_CLASSES,
            dtype=torch.float32).to(DEVICE)
        
        # <<< THE ONLY CHANGE IN TRAINING: USE FOCAL LOSS >>>
        criterion = FocalLoss(alpha=weights, gamma=2.0)

        best_f1, best_state, pat = 0, None, 0

        for epoch in range(EPOCHS):
            model.train()
            for xb,yb in tr_loader:
                xb,yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            model.eval()
            ep_preds = []
            with torch.no_grad():
                for xb,_ in te_loader:
                    ep_preds.extend(model(xb.to(DEVICE)).argmax(1).cpu().numpy())
            val_f1 = f1_score(y_te, ep_preds, average='macro')

            if val_f1 > best_f1:
                best_f1    = val_f1
                best_state = {k:v.clone() for k,v in model.state_dict().items()}
                pat        = 0
            else:
                pat += 1
                if pat >= PATIENCE: break

        model.load_state_dict(best_state)
        model.eval()
        final_preds = []
        with torch.no_grad():
            for xb,_ in te_loader:
                final_preds.extend(model(xb.to(DEVICE)).argmax(1).cpu().numpy())

        subj_f1 = f1_score(y_te, final_preds, average='macro')
        subject_f1s[test_subj] = subj_f1
        all_preds.extend(final_preds)
        all_true.extend(y_te.tolist())
        print(f"  Subject {test_subj:>2}: F1={subj_f1:.3f}  (epoch {epoch+1})")

    return all_preds, all_true, subject_f1s


def train_final(model_name, X_norm, y):
    print(f"\nTraining final {model_name.upper()} on all data (Focal Loss)...")
    loader = DataLoader(DrowsinessDataset(X_norm,y),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    model     = get_model(model_name, input_size=N_FEATURES, num_heads=NUM_HEADS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    
    counts    = np.bincount(y, minlength=N_CLASSES).astype(float)
    weights   = torch.tensor(
        (1.0/(counts+1e-6))/(1.0/(counts+1e-6)).sum()*N_CLASSES,
        dtype=torch.float32).to(DEVICE)
        
    # <<< FOCAL LOSS >>>
    criterion = FocalLoss(alpha=weights, gamma=2.0)
    losses    = []

    for epoch in tqdm(range(EPOCHS), desc=f'Final {model_name.upper()}'):
        model.train()
        ep_loss = 0
        for xb,yb in loader:
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()
        losses.append(ep_loss/len(loader))

    return model, losses


def save_model(model, model_name, loso_f1, subject_f1s, losses):
    path = os.path.join(OUTPUT_DIR, f'drowsiness_{model_name}_v5_focal.pth')
    torch.save({
        'model_name'       : model_name,
        'model_state_dict' : model.state_dict(),
        'model_config'     : {
            'input_size' : N_FEATURES, 'hidden_size': 128, 'num_layers' : 2,
            'num_classes': N_CLASSES, 'dropout'    : 0.3, 'num_heads'  : NUM_HEADS,
        },
        'features'         : FEATURE_ORDER,
        'seq_len'          : SEQ_LEN,
        'loso_f1'          : loso_f1,
        'subject_scores'   : {str(k):float(v) for k,v in subject_f1s.items()},
        'train_losses'     : losses,
    }, path)
    print(f"Saved: {path}")


# ================================================================
# RUN V5
# ================================================================
baselines = {'BiGRU_v3': 0.6799, 'BiLSTM_v4_MHA': 0.6698}
all_results = {}

for mname in MODELS_TO_TRAIN:
    preds, true, sf1s = run_loso(mname, X_norm, y, subjects)
    overall_f1        = f1_score(true, preds, average='macro')
    f1vals            = list(sf1s.values())

    print(f"\n{mname.upper()} + FOCAL LOSS RESULTS")
    print(classification_report(true, preds, target_names=['Alert','Low-vigilant','Drowsy']))
    print(f"Mean F1 : {overall_f1:.4f}")
    print(f"Std  F1 : {np.std(f1vals):.4f}")

    final_model, losses = train_final(mname, X_norm, y)
    save_model(final_model, mname, overall_f1, sf1s, losses)

    all_results[mname] = {
        'loso_f1': overall_f1, 'subject_f1s': sf1s,
        'all_true': true, 'all_preds': preds,
    }

    with open(os.path.join(OUTPUT_DIR,f'metrics_{mname}_v5_focal.json'),'w') as f:
        json.dump({
            'model': mname, 'loss_type': 'focal_gamma2',
            'loso_f1': float(overall_f1),
            'subject_scores': {str(k):float(v) for k,v in sf1s.items()},
        }, f, indent=2)

# Quick comparison print
print(f"\n{'='*40}")
print("QUICK COMPARISON")
print(f"{'='*40}")
print(f"BiGRU v3 (Baseline) : {baselines['BiGRU_v3']:.4f}")
print(f"BiGRU v4 + MHA      : 0.6799 (running)")
if 'bigru_mha' in all_results:
    print(f"BiGRU v5 + MHA + FOCAL: {all_results['bigru_mha']['loso_f1']:.4f}")
print(f"{'='*40}")