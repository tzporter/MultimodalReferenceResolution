import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import os
import json

# Traditional ML imports
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA # <--- Added for Dimension Reduction

# Import your base model12
from base_model import MultimodalPoseAudioTransformer 

# ==========================================
# 1. Setup & Hyperparameters
# ==========================================

POSE_DIM = 159
AUDIO_DIM = 128
MODEL_DIM = 256
NHEAD = 8
NUM_LAYERS = 6
DIM_FEEDFORWARD = 1024
WINDOW_SIZE = 72
STEP_SIZE = WINDOW_SIZE // 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Classification Params ---
NUM_CLASSES = 2
FINETUNE_BATCH_SIZE = 32 # Can be larger now since we aren't running the Transformer
FINETUNE_EPOCHS = 20
FINETUNE_LR = 0.001

# --- Dimension Reduction Params ---
REDUCED_DIMENSION = 16  # Reduce 256 -> 16 for "manageable size"

BEST_CHECKPOINT_PATH = "checkpoints/epoch_34_valloss_1.0583.pth" 

# ==========================================
# 2. Classes (Dataset & Model Helpers)
# ==========================================

class PoseAudioClassificationDataset(Dataset):
    """
    Original Dataset to load raw sequences. 
    Used ONLY during the initial feature extraction phase.
    """
    def __init__(self, pose_sequences, audio_sequences, label_sequences, window_size, step_size):
        self.full_pose_sequences = []
        self.full_audio_sequences = []
        self.full_label_sequences = []
        self.window_indices = []

        for i, (pose_seq, audio_seq, label) in enumerate(zip(pose_sequences, audio_sequences, label_sequences)):
            if isinstance(pose_seq, np.ndarray): pose_seq = torch.from_numpy(pose_seq).float()
            if isinstance(audio_seq, np.ndarray): audio_seq = torch.from_numpy(audio_seq).float()
            
            min_len = min(pose_seq.shape[0], audio_seq.shape[0])
            if min_len < window_size: continue
                
            self.full_pose_sequences.append(pose_seq)
            self.full_audio_sequences.append(audio_seq)
            self.full_label_sequences.append(label)
            
            new_sequence_index = len(self.full_pose_sequences) - 1
            for start_frame in range(0, min_len - window_size + 1, step_size):
                # Ensure label consistency in window
                if label[start_frame] >= 0 and all(label[start_frame] == label[frame] for frame in range(start_frame, start_frame + window_size)):
                    self.window_indices.append((new_sequence_index, start_frame))

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        seq_idx, start = self.window_indices[idx]
        end = start + WINDOW_SIZE
        return (self.full_pose_sequences[seq_idx][start:end], 
                self.full_audio_sequences[seq_idx][start:end], 
                self.full_label_sequences[seq_idx][start])

def create_classification_collate_fn(pad_value: float = 0.0):
    def collate_fn(batch):
        X_poses, X_audios, labels = zip(*batch)
        X_poses_padded = pad_sequence(X_poses, batch_first=True, padding_value=pad_value)
        X_audios_padded = pad_sequence(X_audios, batch_first=True, padding_value=pad_value)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        lengths = [len(x) for x in X_poses]
        max_len = X_poses_padded.shape[1]
        padding_mask = torch.arange(max_len)[None, :] >= torch.tensor(lengths)[:, None]
        return X_poses_padded, X_audios_padded, labels_tensor, padding_mask
    return collate_fn

class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        # Note: input_dim will be REDUCED_DIMENSION (e.g., 64)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), # Scaled up slightly for non-linearity
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)

def load_sequences(chapters):
    # Placeholder for your actual data loading logic
    pose_data, audio_data, label_data = [], [], []
    for chapter in chapters:
        pose_file = f"../data/normalized_poses/poses-{chapter}.npy"
        audio_file = f"../data/audio_features/audio-{chapter}.npy"
        label_file = f"../data/downsampled_labels/labels-{chapter}.npy"
        pose_data.append(np.load(pose_file))
        audio_data.append(np.load(audio_file))
        label_data.append(np.load(label_file))
    return pose_data, audio_data, label_data

# ==========================================
# 3. Execution Logic
# ==========================================

def main():
    # --- A. Load Raw Data ---
    print("Loading raw data...")
    with open("train_test_chapters.json", "r") as f:
        ch_dict = json.load(f)
    train_pose, train_audio, train_labels = load_sequences(ch_dict['train'])
    val_pose, val_audio, val_labels = load_sequences(ch_dict['test'])

    # --- B. Initialize Base Model (Frozen) ---
    print(f"Loading base model from {BEST_CHECKPOINT_PATH}...")
    # Note: Ensure MultimodalPoseAudioTransformer class is available
    base_model = MultimodalPoseAudioTransformer(
        pose_dim=POSE_DIM, audio_dim=AUDIO_DIM, model_dim=MODEL_DIM,
        nhead=NHEAD, num_encoder_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
        max_seq_len=WINDOW_SIZE + 1
    ).to(DEVICE)
    
    checkpoint = torch.load(BEST_CHECKPOINT_PATH, map_location=DEVICE)
    base_model.load_state_dict(checkpoint['model_state_dict'])
    base_model.eval()
    
    # Freeze parameters to save memory/compute during extraction
    for param in base_model.parameters():
        param.requires_grad = False

    # --- C. Define Feature Extraction Function ---
    def extract_embeddings(dataset, model, device, batch_size=32):
        """Generates embeddings ONCE for the entire dataset."""
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                            collate_fn=create_classification_collate_fn())
        
        all_embeddings = []
        all_labels = []
        
        print(f"Generating embeddings for {len(dataset)} samples...")
        
        with torch.no_grad():
            for X_pose, X_audio, labels, padding_mask in loader:
                X_pose, X_audio = X_pose.to(device), X_audio.to(device)
                padding_mask = padding_mask.to(device)
                
                # Create masks
                seq_len = X_pose.shape[1]
                pose_causal = nn.Transformer.generate_square_subsequent_mask(seq_len + 1).to(device)
                audio_causal = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
                cls_pad = torch.zeros(X_pose.shape[0], 1, dtype=torch.bool, device=device)
                pose_pad = torch.cat([cls_pad, padding_mask], dim=1)
                
                # Forward pass
                _, _, embeddings = model(
                    pose_seq=X_pose, audio_seq=X_audio,
                    pose_mask=pose_causal, audio_mask=audio_causal,
                    pose_padding_mask=pose_pad, audio_padding_mask=padding_mask
                )
                
                all_embeddings.append(embeddings.cpu().numpy())
                all_labels.append(labels.numpy())
                
        return np.concatenate(all_embeddings), np.concatenate(all_labels)

    # --- D. Generate Embeddings (The "Once" Step) ---
    train_ds_raw = PoseAudioClassificationDataset(train_pose, train_audio, train_labels, WINDOW_SIZE, STEP_SIZE)
    val_ds_raw = PoseAudioClassificationDataset(val_pose, val_audio, val_labels, WINDOW_SIZE, STEP_SIZE)

    print("\n--- Phase 1: Extracting Embeddings ---", flush=True)
    X_train_full, y_train = extract_embeddings(train_ds_raw, base_model, DEVICE)
    X_val_full, y_val = extract_embeddings(val_ds_raw, base_model, DEVICE)

    # We can now unload the base model to free up GPU memory
    del base_model
    torch.cuda.empty_cache() 
    print("Base model unloaded from GPU.")

    # --- E. Debug Info & Content Analysis ---
    print("\n--- Phase 2: Debugging & Analysis of Embeddings ---", flush=True)
    print(f"Original Embedding Shape: {X_train_full.shape}")
    
    # 1. Basic Statistics
    print(f"Mean value: {np.mean(X_train_full):.4f}")
    print(f"Std Dev:    {np.std(X_train_full):.4f}")
    print(f"Max value:  {np.max(X_train_full):.4f}")
    print(f"Min value:  {np.min(X_train_full):.4f}")
    
    # 2. Sparsity Check (ReLU models often produce sparse activations)
    zero_count = np.sum(np.isclose(X_train_full, 0, atol=1e-5))
    total_params = X_train_full.size
    print(f"Sparsity (approx % zeros): {(zero_count/total_params)*100:.2f}%")
    
    # 3. Norm Check (Did LayerNorm work?)
    norms = np.linalg.norm(X_train_full, axis=1)
    print(f"Average Vector Norm: {np.mean(norms):.4f} (Should be close to sqrt(d) or stable)")

    # --- F. Dimensionality Reduction ---
    print("\n--- Phase 3: Dimensionality Reduction (PCA) ---", flush=True)
    print(f"Reducing dimensions from {MODEL_DIM} to {REDUCED_DIMENSION}...")
    
    pca = PCA(n_components=REDUCED_DIMENSION, random_state=42)
    
    # Fit ONLY on training data, transform both
    X_train_reduced = pca.fit_transform(X_train_full)
    X_val_reduced = pca.transform(X_val_full)
    
    explained_var = np.sum(pca.explained_variance_ratio_)
    print(f"PCA Explained Variance: {explained_var*100:.2f}% of original information retained.")
    print(f"New Training Shape: {X_train_reduced.shape}")

    # --- G. Prepare Optimized DataLoaders ---
    # Now we use simple TensorDatasets because the data is just vectors
    train_ds_opt = TensorDataset(torch.tensor(X_train_reduced, dtype=torch.float32), 
                                 torch.tensor(y_train, dtype=torch.long))
    val_ds_opt = TensorDataset(torch.tensor(X_val_reduced, dtype=torch.float32), 
                               torch.tensor(y_val, dtype=torch.long))
    
    train_loader_opt = DataLoader(train_ds_opt, batch_size=FINETUNE_BATCH_SIZE, shuffle=True)
    val_loader_opt = DataLoader(val_ds_opt, batch_size=FINETUNE_BATCH_SIZE, shuffle=False)

    # --- H. Train MLP Head (Fine-tuning replacement) ---
    print("\n--- Phase 4: Training Classification Head on Reduced Embeddings ---", flush=True)
    
    classifier = ClassificationHead(input_dim=REDUCED_DIMENSION, num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=FINETUNE_LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, FINETUNE_EPOCHS + 1):
        classifier.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for emb_batch, label_batch in train_loader_opt:
            emb_batch, label_batch = emb_batch.to(DEVICE), label_batch.to(DEVICE)
            
            optimizer.zero_grad()
            logits = classifier(emb_batch)
            loss = criterion(logits, label_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == label_batch).sum().item()
            train_total += label_batch.size(0)

        # Validation
        classifier.eval()
        val_correct, val_total = 0, 0
        nn_preds_list = []
        
        with torch.no_grad():
            for emb_batch, label_batch in val_loader_opt:
                emb_batch, label_batch = emb_batch.to(DEVICE), label_batch.to(DEVICE)
                logits = classifier(emb_batch)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == label_batch).sum().item()
                val_total += label_batch.size(0)
                nn_preds_list.append(preds.cpu().numpy())

        print(f"Epoch {epoch} | Train Acc: {(train_correct/train_total)*100:.2f}% | Val Acc: {(val_correct/val_total)*100:.2f}%", flush=True)

    nn_predictions = np.concatenate(nn_preds_list)
    
    # Get final training predictions for NN
    nn_train_preds_list = []
    classifier.eval()
    with torch.no_grad():
        for emb_batch, _ in train_loader_opt:
            emb_batch = emb_batch.to(DEVICE)
            logits = classifier(emb_batch)
            preds = torch.argmax(logits, dim=1)
            nn_train_preds_list.append(preds.cpu().numpy())
    nn_train_predictions = np.concatenate(nn_train_preds_list)

    # --- I. Train Traditional ML Models ---
    print("\n--- Phase 5: Training Traditional Models on Reduced Embeddings ---", flush=True)
    
    # SVM
    print("Training SVM...")
    svc = SVC(kernel='rbf', gamma='scale')
    svc.fit(X_train_reduced, y_train)
    svc_train_preds = svc.predict(X_train_reduced)
    svc_preds = svc.predict(X_val_reduced)
    
    # Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    rf.fit(X_train_reduced, y_train)
    rf_train_preds = rf.predict(X_train_reduced)
    rf_preds = rf.predict(X_val_reduced)

    # --- J. Final Comparison ---
    print("\n" + "="*80)
    print("FINAL RESULTS (on PCA-reduced Embeddings)")
    print("="*80)
    
    models = {
        "Neural Net Head": (nn_train_predictions, nn_predictions),
        "SVM (RBF)": (svc_train_preds, svc_preds),
        "Random Forest": (rf_train_preds, rf_preds)
    }
    
    print(f"{'Model':<20} {'Train Acc':<15} {'Val Acc':<15} {'Precision':<15} {'Recall':<15}")
    print("-" * 80)
    
    results_dict = {}
    
    for name, (train_preds, val_preds) in models.items():
        train_acc = accuracy_score(y_train, train_preds)
        val_acc = accuracy_score(y_val, val_preds)
        prec = precision_score(y_val, val_preds, average='weighted', zero_division=0)
        rec = recall_score(y_val, val_preds, average='weighted', zero_division=0)
        print(f"{name:<20} {train_acc:.4f} ({train_acc*100:5.2f}%)  {val_acc:.4f} ({val_acc*100:5.2f}%)  {prec:.4f}           {rec:.4f}")
        
        results_dict[name] = {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'precision': prec, 
            'recall': rec, 
            'report': classification_report(y_val, val_preds, output_dict=True)
        }

if __name__ == "__main__":
    main()