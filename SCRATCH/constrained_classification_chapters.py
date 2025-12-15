import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
import os
import json

# ML Imports
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

# Import your base model definition
from base_model import MultimodalPoseAudioTransformer

# ==========================================
# 1. Hyperparameters & Setup
# ==========================================
POSE_DIM = 159
AUDIO_DIM = 128
MODEL_DIM = 256
NHEAD = 8
NUM_LAYERS = 6
DIM_FEEDFORWARD = 1024
WINDOW_SIZE = 72
STEP_SIZE = WINDOW_SIZE // 4  # Used only if we needed to generate sliding windows (we use CSV instead)
FPS = 24                      # Needed for CSV timestamp conversion

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Classification & Reduction Params ---
NUM_CLASSES = 2
REDUCED_DIMENSION = 16        # Reduce 256 -> 16 via PCA
FINETUNE_BATCH_SIZE = 32
FINETUNE_EPOCHS = 20
FINETUNE_LR = 0.0005

# --- Paths ---
# Using the checkpoint from your requested code
BEST_CHECKPOINT_PATH = "checkpoints/epoch_34_valloss_1.0583.pth"
SPEAKER_SEGMENTS_PATH = "../speaker_segments.csv"
CHAPTERS_JSON_PATH = "train_test_chapters.json"

# ==========================================
# 2. Dataset & Helpers
# ==========================================

class RawExtractionDataset(Dataset):
    """
    Dataset used ONLY for the initial pass to extract embeddings from the base model.
    It accepts pre-calculated window_indices derived from your CSV.
    """
    def __init__(self, pose_data, audio_data, window_indices, labels, window_size):
        self.pose_data = [torch.from_numpy(p).float() if isinstance(p, np.ndarray) else p for p in pose_data]
        self.audio_data = [torch.from_numpy(a).float() if isinstance(a, np.ndarray) else a for a in audio_data]
        self.window_indices = window_indices
        self.labels = labels
        self.window_size = window_size

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        seq_idx, start_frame = self.window_indices[idx]
        end_frame = start_frame + self.window_size
        
        full_pose = self.pose_data[seq_idx]
        full_audio = self.audio_data[seq_idx]
        
        # Safety bounds check
        if end_frame > len(full_pose) or end_frame > len(full_audio):
            end_frame = min(len(full_pose), len(full_audio))
            start_frame = max(0, end_frame - self.window_size)

        return (full_pose[start_frame:end_frame], 
                full_audio[start_frame:end_frame], 
                self.labels[idx])

def extraction_collate_fn(batch):
    """Pads sequences if they are shorter than window_size (edge cases)."""
    X_p, X_a, y = zip(*batch)
    X_p_pad = pad_sequence(X_p, batch_first=True, padding_value=0.0)
    X_a_pad = pad_sequence(X_a, batch_first=True, padding_value=0.0)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    # Create mask
    lengths = [len(x) for x in X_p]
    max_len = X_p_pad.shape[1]
    mask = torch.arange(max_len)[None, :] >= torch.tensor(lengths)[:, None]
    
    return X_p_pad, X_a_pad, y_tensor, mask

class ClassificationHead(nn.Module):
    """Simple MLP to run on top of the PCA-reduced embeddings."""
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.net(x)

def load_raw_files(chapters):
    """Loads the .npy files for the given chapters."""
    p_data, a_data = [], []
    for ch in chapters:
        p_data.append(np.load(f"../data/normalized_poses/poses-{ch}.npy"))
        a_data.append(np.load(f"../data/audio_features/audio-{ch}.npy"))
    return p_data, a_data

def parse_csv_windows(chapters, segments_df, fps):
    """
    Parses the CSV to create (sequence_index, start_frame) tuples.
    """
    window_indices = []
    labels = []
    
    for i, chapter in enumerate(chapters):
        # Filter CSV for this chapter
        chapter_segs = segments_df[segments_df['pair_speaker'] == chapter]
        
        for _, row in chapter_segs.iterrows():
            start_frame = int(row['from_ts'] * fps)
            label = int(row['is_present'])
            
            window_indices.append((i, start_frame))
            labels.append(label)
            
    return window_indices, labels

# ==========================================
# 3. Main Execution
# ==========================================

def main():
    # --- Step 1: Load Data ---
    print("Loading configuration and raw data...")
    with open(CHAPTERS_JSON_PATH, "r") as f:
        ch_dict = json.load(f)
    
    train_pose, train_audio = load_raw_files(ch_dict['train'])
    val_pose, val_audio = load_raw_files(ch_dict['test'])
    
    print(f"Loaded {len(train_pose)} train sequences and {len(val_pose)} test sequences.")

    # --- Step 2: Parse CSV for Windows ---
    print("Parsing Speaker Segments CSV...")
    segments_df = pd.read_csv(SPEAKER_SEGMENTS_PATH)
    
    train_indices, train_labels = parse_csv_windows(ch_dict['train'], segments_df, FPS)
    val_indices, val_labels = parse_csv_windows(ch_dict['test'], segments_df, FPS)
    
    print(f"Defined {len(train_indices)} training windows and {len(val_indices)} validation windows based on CSV.")

    # --- Step 3: Initialize Extraction Datasets ---
    train_ds_raw = RawExtractionDataset(train_pose, train_audio, train_indices, train_labels, WINDOW_SIZE)
    val_ds_raw = RawExtractionDataset(val_pose, val_audio, val_indices, val_labels, WINDOW_SIZE)

    # --- Step 4: Load Base Model (Frozen) ---
    print(f"Loading base model from {BEST_CHECKPOINT_PATH}...", flush=True)
    base_model = MultimodalPoseAudioTransformer(
        pose_dim=POSE_DIM, audio_dim=AUDIO_DIM, model_dim=MODEL_DIM,
        nhead=NHEAD, num_encoder_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
        max_seq_len=WINDOW_SIZE + 1
    ).to(DEVICE)
    
    checkpoint = torch.load(BEST_CHECKPOINT_PATH, map_location=DEVICE)
    base_model.load_state_dict(checkpoint['model_state_dict'])
    base_model.eval()
    for p in base_model.parameters(): p.requires_grad = False

    # --- Step 5: One-Time Feature Extraction ---
    def extract(dataset, model):
        loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=extraction_collate_fn)
        embs, lbls = [], []
        print(f"Extracting features for {len(dataset)} samples...")
        
        with torch.no_grad():
            for X_p, X_a, y, mask in loader:
                X_p, X_a, mask = X_p.to(DEVICE), X_a.to(DEVICE), mask.to(DEVICE)
                
                # Masks for Transformer
                seq_len = X_p.shape[1]
                p_causal = nn.Transformer.generate_square_subsequent_mask(seq_len + 1).to(DEVICE)
                a_causal = nn.Transformer.generate_square_subsequent_mask(seq_len).to(DEVICE)
                cls_pad = torch.zeros(X_p.shape[0], 1, dtype=torch.bool, device=DEVICE)
                p_pad = torch.cat([cls_pad, mask], dim=1)
                
                _, _, embedding = model(
                    pose_seq=X_p, audio_seq=X_a, 
                    pose_mask=p_causal, audio_mask=a_causal,
                    pose_padding_mask=p_pad, audio_padding_mask=mask
                )
                embs.append(embedding.cpu().numpy())
                lbls.append(y.numpy())
        return np.concatenate(embs), np.concatenate(lbls)

    print("\n--- Phase 1: Feature Extraction ---", flush=True)
    X_train_full, y_train = extract(train_ds_raw, base_model)
    X_val_full, y_val = extract(val_ds_raw, base_model)

    # Free up GPU
    del base_model
    torch.cuda.empty_cache()
    print("Base model unloaded.")

    # --- Step 6: Dimensionality Reduction (PCA) ---
    print("\n--- Phase 2: PCA Reduction ---", flush=True)
    # Check for NaNs/Infs
    if np.any(np.isnan(X_train_full)) or np.any(np.isinf(X_train_full)):
        print("Warning: NaNs or Infs detected in embeddings. Replacing with 0.")
        X_train_full = np.nan_to_num(X_train_full)
        X_val_full = np.nan_to_num(X_val_full)

    pca = PCA(n_components=REDUCED_DIMENSION, random_state=42)
    X_train_red = pca.fit_transform(X_train_full)
    X_val_red = pca.transform(X_val_full)
    
    print(f"Reduced shape: {X_train_red.shape} | Variance Retained: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")

    # --- Step 7: Prepare Fast DataLoaders ---
    train_ds_opt = TensorDataset(torch.tensor(X_train_red, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds_opt = TensorDataset(torch.tensor(X_val_red, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    
    train_loader = DataLoader(train_ds_opt, batch_size=FINETUNE_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds_opt, batch_size=FINETUNE_BATCH_SIZE, shuffle=False)

    # --- Step 8: Model Training & Comparison ---
    print("\n--- Phase 3: Model Training ---", flush=True)

    # 8a. Neural Network Head
    print("Training Neural Network...", flush=True)
    clf = ClassificationHead(REDUCED_DIMENSION, NUM_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(clf.parameters(), lr=FINETUNE_LR)
    crit = nn.CrossEntropyLoss()
    
    for ep in range(1, FINETUNE_EPOCHS+1):
        clf.train()
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            opt.zero_grad()
            loss = crit(clf(bx), by)
            loss.backward()
            opt.step()
            
    # Eval NN
    clf.eval()
    nn_preds = []
    nn_train_preds = []
    with torch.no_grad():
        for bx, _ in train_loader:
            logits = clf(bx.to(DEVICE))
            nn_train_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        for bx, _ in val_loader:
            logits = clf(bx.to(DEVICE))
            nn_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
    nn_train_preds = np.concatenate(nn_train_preds)
    nn_preds = np.concatenate(nn_preds)

    # 8b. SVM
    print("Training SVM...")
    svc = SVC(kernel='rbf', gamma='scale', random_state=42)
    svc.fit(X_train_red, y_train)
    svc_train_preds = svc.predict(X_train_red)
    svc_preds = svc.predict(X_val_red)

    # 8c. Random Forest
    print("Training Random Forest...", flush=True)
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X_train_red, y_train)
    rf_train_preds = rf.predict(X_train_red)
    rf_preds = rf.predict(X_val_red)

    # --- Step 9: Results ---
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    print(f"{'Model':<20} {'Train Acc':<15} {'Val Acc':<15} {'Precision':<15} {'Recall':<15}")
    print("-"*80)
    
    results_dict = {}
    models = [
        ("Neural Network", nn_train_preds, nn_preds), 
        ("SVM", svc_train_preds, svc_preds), 
        ("Random Forest", rf_train_preds, rf_preds)
    ]
    
    for name, train_preds, val_preds in models:
        train_acc = accuracy_score(y_train, train_preds)
        val_acc = accuracy_score(y_val, val_preds)
        prec = precision_score(y_val, val_preds, average='weighted', zero_division=0)
        rec = recall_score(y_val, val_preds, average='weighted', zero_division=0)
        print(f"{name:<20} {train_acc:.4f} ({train_acc*100:5.2f}%)  {val_acc:.4f} ({val_acc*100:5.2f}%)  {prec:.4f}           {rec:.4f}")
        
        results_dict[name.lower().replace(" ", "_")] = {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'precision': prec, 
            'recall': rec, 
            'confusion_matrix': confusion_matrix(y_val, val_preds).tolist()
        }

    with open('model_comparison_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    print("\nSaved results to model_comparison_results.json")

if __name__ == "__main__":
    main()