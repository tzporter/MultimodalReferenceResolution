import torch
import torch.nn as nn
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from base_model import MultimodalPoseAudioTransformer
import json



# ==================================================================
# --- MODIFIED PoseAudioSequenceDataset CLASS ---
# ==================================================================

class PoseAudioSequenceDataset(Dataset):
    """
    A dataset that returns *windows* of sequences for next-step prediction.
    
    Instead of returning one full (truncated) sequence, this dataset
    breaks long sequences into overlapping windows.
    """
    def __init__(self, pose_sequences: list, audio_sequences: list, 
                 window_size: int, step_size: int):
        """
        Args:
            pose_sequences: A list of *full* Tensors [seq_len, pose_dim]
            audio_sequences: A list of *full* Tensors [seq_len, audio_dim]
            window_size: The length of the subsequences (e.g., 1000 frames)
            step_size: The step (stride) between windows (e.g., 250 frames)
        """
        assert len(pose_sequences) == len(audio_sequences)
        
        self.full_pose_sequences = []
        self.full_audio_sequences = []
        self.window_size = window_size
        self.step_size = step_size
        
        # This list will store (seq_index, start_frame) tuples
        self.window_indices = []

        # The total number of frames needed to get an (X, Y) pair
        # We need window_size frames for X and 1 extra for Y's last frame
        self.total_frames_needed = self.window_size + 1 

        for i, (pose_seq, audio_seq) in enumerate(zip(pose_sequences, audio_sequences)):
            # Convert to tensor if they are numpy arrays
            if isinstance(pose_seq, np.ndarray):
                pose_seq = torch.from_numpy(pose_seq).float()
            if isinstance(audio_seq, np.ndarray):
                audio_seq = torch.from_numpy(audio_seq).float()

            # Find the shortest length to align them
            min_len = min(pose_seq.shape[0], audio_seq.shape[0])
            
            # Skip if the sequence is too short to make even one window
            if min_len < self.total_frames_needed:
                continue
                
            self.full_pose_sequences.append(pose_seq)
            self.full_audio_sequences.append(audio_seq)
            
            # Get the *actual* index in our new lists
            new_sequence_index = len(self.full_pose_sequences) - 1
            
            # Create window indices for this sequence
            # We slide from 0 up to the last possible start frame
            for start_frame in range(0, min_len - self.total_frames_needed + 1, self.step_size):
                self.window_indices.append((new_sequence_index, start_frame))

    def __len__(self):
        # The length is the total number of windows we can make
        return len(self.window_indices)

    def __getitem__(self, idx):
        # Get the pointer for this window
        sequence_index, start_frame = self.window_indices[idx]
        
        # Get the full sequences
        full_pose_seq = self.full_pose_sequences[sequence_index]
        full_audio_seq = self.full_audio_sequences[sequence_index]
        
        # Calculate the end frame
        # We slice [start : start + window_size + 1]
        end_frame = start_frame + self.total_frames_needed
        
        # Slice the full sequence to get our chunk
        pose_chunk = full_pose_seq[start_frame:end_frame]
        audio_chunk = full_audio_seq[start_frame:end_frame]
        
        # Create the X (input) and Y (target) pairs
        # X is frames 0 to (window_size - 1)
        X_pose = pose_chunk[:-1]
        X_audio = audio_chunk[:-1]
        
        # Y is frames 1 to window_size
        Y_pose = pose_chunk[1:]
        Y_audio = audio_chunk[1:]
        
        # X_pose, X_audio, Y_pose, Y_audio will all have length = self.window_size
        return X_pose, X_audio, Y_pose, Y_audio

# --- [create_collate_fn function] ---
# (No changes needed, this will still work!)
def create_collate_fn(pad_value: float = 0.0):
    def collate_fn(batch):
        X_poses, X_audios, Y_poses, Y_audios = zip(*batch)
        
        # Since all items from our new Dataset have the same length (window_size),
        # this padding step is trivial but harmless and correct.
        X_poses_padded = pad_sequence(X_poses, batch_first=True, padding_value=pad_value)
        X_audios_padded = pad_sequence(X_audios, batch_first=True, padding_value=pad_value)
        Y_poses_padded = pad_sequence(Y_poses, batch_first=True, padding_value=pad_value)
        Y_audios_padded = pad_sequence(Y_audios, batch_first=True, padding_value=pad_value)
        
        max_len = X_poses_padded.shape[1]
        lengths = [len(x) for x in X_poses]
        lengths_tensor = torch.tensor(lengths)
        
        # This mask will be all 'False' (no padding), which is correct.
        padding_mask = torch.arange(max_len)[None, :] >= lengths_tensor[:, None]
        
        return X_poses_padded, X_audios_padded, Y_poses_padded, Y_audios_padded, padding_mask

    return collate_fn

# --- [load_sequences function] ---
# (No changes needed)
def load_sequences(chapters):
    pose_data = []
    audio_data = []
    
    for chapter in chapters:
        pose_file = f"../data/normalized_poses/poses-{chapter}.npy"
        audio_file = f"../data/audio_features/audio-{chapter}.npy"
        # We load as numpy first, conversion to tensor happens in Dataset
        pose_data.append(np.load(pose_file))
        audio_data.append(np.load(audio_file))
    return pose_data, audio_data

# ==================================================================
# --- MODIFIED SCRIPT STARTS HERE ---
# ==================================================================

# === 1. Hyperparameters ===
POSE_DIM = 159
AUDIO_DIM = 128
MODEL_DIM = 256
NHEAD = 8
NUM_LAYERS = 6
DIM_FEEDFORWARD = 1024
WINDOW_SIZE = 72     # <-- RENAMED from MAX_LEN
STEP_SIZE = WINDOW_SIZE // 4 # <-- ADDED (75% overlap, e.g., 250)
BATCH_SIZE = 32
NUM_EPOCHS = 100
CHECKPOINT_DIR = "checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_BF16 = True  # Set to False to disable
if USE_BF16 and torch.cuda.is_available():
    # Check if your GPU supports BF16
    if torch.cuda.is_bf16_supported():
        print("BF16 training enabled")
    else:
        print("Warning: BF16 not supported on this GPU, falling back to FP32")
        USE_BF16 = False
else:
    USE_BF16 = False




print(f"Using device: {DEVICE}")
print(f"Window Size: {WINDOW_SIZE}, Step Size: {STEP_SIZE}")

# === 2. Load and Split Data ===
print("Loading data...")
with open("train_test_chapters.json", "r") as f:
    ch_dict = json.load(f)
train_pose, train_audio = load_sequences(ch_dict['train'])
val_pose, val_audio = load_sequences(ch_dict['test'])
print(f"Training sequences: {len(train_pose)}")
print(f"Validation sequences: {len(val_pose)}")


# === 3. Setup DataLoaders ===
custom_collate_fn = create_collate_fn(pad_value=0.0)

# <-- MODIFIED: Pass window_size and step_size to Dataset
train_dataset = PoseAudioSequenceDataset(train_pose, train_audio, WINDOW_SIZE, STEP_SIZE)
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    collate_fn=custom_collate_fn,
    num_workers=4,
    pin_memory=True # <-- Optional: for faster CPU->GPU transfer
)

val_dataset = PoseAudioSequenceDataset(val_pose, val_audio, WINDOW_SIZE, STEP_SIZE)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=custom_collate_fn,
    num_workers=4,
    pin_memory=True # <-- Optional
)

# Print total number of windows
print(f"Total training windows: {len(train_dataset)}")
print(f"Total validation windows: {len(val_dataset)}")

# === 4. Initialize Model and Optimizer ===
model = MultimodalPoseAudioTransformer(
    pose_dim=POSE_DIM,
    audio_dim=AUDIO_DIM,
    model_dim=MODEL_DIM,
    nhead=NHEAD,
    num_encoder_layers=NUM_LAYERS,
    dim_feedforward=DIM_FEEDFORWARD,
    # Pass WINDOW_SIZE + 1 to account for CLS token
    max_seq_len=WINDOW_SIZE + 1 
).to(DEVICE)

pose_loss_fn = nn.MSELoss(reduction='none')
audio_loss_fn = nn.MSELoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# === 5. Training and Validation Loop ===

# --- [Checkpointing functions] ---
# (No changes needed)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
best_checkpoints = [(float('inf'), None) for _ in range(3)]
def save_checkpoint(epoch, model, optimizer, loss, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': loss
    }
    torch.save(checkpoint, path)
def update_best_checkpoints(epoch, model, optimizer, val_loss, best_list):
    if val_loss < best_list[-1][0]:
        worst_loss, worst_path = best_list.pop()
        if worst_path is not None:
            try:
                os.remove(worst_path)
                print(f"Removed old checkpoint: {worst_path} (Loss: {worst_loss:.4f})")
            except OSError as e:
                print(f"Error removing file {worst_path}: {e}")
        new_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}_valloss_{val_loss:.4f}.pth")
        save_checkpoint(epoch, model, optimizer, val_loss, new_path)
        best_list.append((val_loss, new_path))
        best_list.sort(key=lambda x: x[0])
        print(f"New top-3 model saved: {new_path} (Loss: {val_loss:.4f})")
# --- [End Checkpointing functions] ---


print("Starting training...")
step = 0
for epoch in range(1, NUM_EPOCHS + 1):
    
    # --- Training Phase ---
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        # Get data and move to device
        X_pose, X_audio, Y_pose, Y_audio, padding_mask = [d.to(DEVICE) for d in batch]
        
        batch_size, seq_len = X_pose.shape[0], X_pose.shape[1]
        assert seq_len == WINDOW_SIZE

        # --- Create Masks ---
        pose_causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len + 1).to(DEVICE)
        audio_causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(DEVICE)
        cls_padding = torch.zeros(batch_size, 1, dtype=torch.bool, device=DEVICE)
        pose_padding_mask = torch.cat([cls_padding, padding_mask], dim=1)
        audio_padding_mask = padding_mask

        # --- Forward Pass with BF16 ---
        # Use autocast context manager for mixed precision
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=USE_BF16):
            pred_pose, pred_audio, _ = model(
                pose_seq=X_pose,
                audio_seq=X_audio,
                pose_mask=pose_causal_mask,
                audio_mask=audio_causal_mask,
                pose_padding_mask=pose_padding_mask,
                audio_padding_mask=audio_padding_mask
            )
            
            # --- Compute Masked Loss ---
            loss_mask = ~padding_mask 
            
            # Pose Loss
            loss_pose_unreduced = pose_loss_fn(pred_pose, Y_pose)
            loss_pose_masked = loss_pose_unreduced * loss_mask.unsqueeze(-1)
            num_real_elements_pose = loss_mask.sum() * (POSE_DIM if POSE_DIM > 0 else 1)
            loss_pose = loss_pose_masked.sum() / num_real_elements_pose

            # Audio Loss
            loss_audio_unreduced = audio_loss_fn(pred_audio, Y_audio)
            loss_audio_masked = loss_audio_unreduced * loss_mask.unsqueeze(-1)
            num_real_elements_audio = loss_mask.sum() * (AUDIO_DIM if AUDIO_DIM > 0 else 1)
            loss_audio = loss_audio_masked.sum() / num_real_elements_audio

            total_loss = loss_pose + loss_audio
        
        # --- Backward Pass ---
        optimizer.zero_grad()
        
        # For BF16, you typically don't need gradient scaling
        # But we keep it here for compatibility (it's disabled above)
        total_loss.backward()
        optimizer.step()
        
        train_loss += total_loss.item()
        if step % 2000 == 0:
            print("step" + str(step) + " train loss: " + str(total_loss.item()), flush=True)
        step += 1
    
    avg_train_loss = train_loss / len(train_loader)

    # --- Validation Phase ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            X_pose, X_audio, Y_pose, Y_audio, padding_mask = [d.to(DEVICE) for d in batch]
            batch_size, seq_len = X_pose.shape[0], X_pose.shape[1]
            assert seq_len == WINDOW_SIZE
            
            pose_causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len + 1).to(DEVICE)
            audio_causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(DEVICE)
            cls_padding = torch.zeros(batch_size, 1, dtype=torch.bool, device=DEVICE)
            pose_padding_mask = torch.cat([cls_padding, padding_mask], dim=1)
            audio_padding_mask = padding_mask

            # Use autocast for validation too
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=USE_BF16):
                pred_pose, pred_audio, _ = model(
                    pose_seq=X_pose,
                    audio_seq=X_audio,
                    pose_mask=pose_causal_mask,
                    audio_mask=audio_causal_mask,
                    pose_padding_mask=pose_padding_mask,
                    audio_padding_mask=audio_padding_mask
                )
                
                loss_mask = ~padding_mask
                
                loss_pose_unreduced = pose_loss_fn(pred_pose, Y_pose)
                loss_pose_masked = loss_pose_unreduced * loss_mask.unsqueeze(-1)
                num_real_elements_pose = loss_mask.sum() * (POSE_DIM if POSE_DIM > 0 else 1)
                loss_pose = loss_pose_masked.sum() / num_real_elements_pose

                loss_audio_unreduced = audio_loss_fn(pred_audio, Y_audio)
                loss_audio_masked = loss_audio_unreduced * loss_mask.unsqueeze(-1)
                num_real_elements_audio = loss_mask.sum() * (AUDIO_DIM if AUDIO_DIM > 0 else 1)
                loss_audio = loss_audio_masked.sum() / num_real_elements_audio

                total_loss = loss_pose + loss_audio
            
            val_loss += total_loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"")
    print(f"Epoch {epoch}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    
    # --- Checkpoint Saving Logic ---
    update_best_checkpoints(epoch, model, optimizer, avg_val_loss, best_checkpoints)

print("\nTraining complete.")
print("Top 3 models saved:")
for loss, path in best_checkpoints:
    if path:
        print(f"  Loss: {loss:.4f}, Path: {path}")