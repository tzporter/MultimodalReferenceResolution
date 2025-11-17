import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    From: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a positional encoding matrix of shape [max_len, d_model]
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        
        # Apply sin to even indices, cos to odd indices
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        # Register as a buffer, not a model parameter
        # Shape: [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        # Add positional encoding
        # self.pe is [max_len, 1, d_model]
        # self.pe.transpose(0, 1) is [1, max_len, d_model]
        # x is [batch_size, seq_len, d_model]
        # We take the first 'seq_len' encodings and add them
        x = x + self.pe.transpose(0, 1)[:, :x.size(1)]
        return self.dropout(x)

class MultimodalPoseAudioTransformer(nn.Module):
    def __init__(self, 
                 pose_dim: int, 
                 audio_dim: int, 
                 model_dim: int, 
                 nhead: int, 
                 num_encoder_layers: int, 
                 dim_feedforward: int, 
                 max_seq_len: int = 1000, 
                 dropout: float = 0.1):
        """
        Args:
            pose_dim: Dimensionality of a single pose keypoint vector (e.g., 50 keypoints * 3D = 150)
            audio_dim: Dimensionality of a single audio feature vector (e.g., 128 MFCCs)
            model_dim: The main dimension of the transformer (must be divisible by nhead)
            nhead: Number of attention heads
            num_encoder_layers: Number of layers for each modality's encoder
            dim_feedforward: Dimension of the feedforward network
            max_seq_len: Maximum possible sequence length (for positional encoding)
            dropout: Dropout rate
        """
        super().__init__()
        self.model_dim = model_dim

        # === 1. Input Embeddings ===
        # Linear layers to project input features to the model dimension
        self.pose_embed = nn.Linear(pose_dim, model_dim)
        self.audio_embed = nn.Linear(audio_dim, model_dim)
        
        # Learnable [CLS] token for classification embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim))
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(model_dim, dropout, max_seq_len)

        # === 2. Modality-Specific Encoders ===
        # Pose Encoder
        pose_encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=nhead, dim_feedforward=dim_feedforward, 
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder_pose = nn.TransformerEncoder(
            pose_encoder_layer, num_layers=num_encoder_layers
        )
        
        # Audio Encoder
        audio_encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=nhead, dim_feedforward=dim_feedforward, 
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder_audio = nn.TransformerEncoder(
            audio_encoder_layer, num_layers=num_encoder_layers
        )

        # === 3. Cross-Attention Fusion ===
        # This layer attends to audio (K, V) from the perspective of pose (Q)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=model_dim, num_heads=nhead, dropout=dropout, batch_first=True
        )
        
        # Standard post-fusion processing block (Add & Norm, FeedForward)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, model_dim)
        )
        self.dropout_ffn = nn.Dropout(dropout)

        # === 4. Prediction Heads ===
        # Linear layers to project back to the original feature dimensions
        self.pose_head = nn.Linear(model_dim, pose_dim)
        self.audio_head = nn.Linear(model_dim, audio_dim)

    def forward(self, 
                pose_seq: torch.Tensor, 
                audio_seq: torch.Tensor, 
                pose_mask: torch.Tensor, 
                audio_mask: torch.Tensor,
                pose_padding_mask: torch.Tensor,
                audio_padding_mask: torch.Tensor):
        """
        Args:
            pose_seq: Pose data, shape [batch, seq_len, pose_dim]
            audio_seq: Audio data, shape [batch, seq_len, audio_dim]
            pose_mask: Causal mask for pose self-attention, shape [seq_len+1, seq_len+1]
            audio_mask: Causal mask for audio self-attention, shape [seq_len, seq_len]
            pose_padding_mask: Padding mask for pose, shape [batch, seq_len+1]
            audio_padding_mask: Padding mask for audio, shape [batch, seq_len]
        """
        
        # --- 1. Pose Encoding ---
        batch_size = pose_seq.shape[0]
        
        # Prepend [CLS] token: [B, 1, D]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        # [B, S, P_dim] -> [B, S, M_dim]
        pose_embedded = self.pose_embed(pose_seq) 
        
        # [B, 1, M_dim] + [B, S, M_dim] -> [B, 1+S, M_dim]
        pose_embedded = torch.cat([cls_tokens, pose_embedded], dim=1)
        
        # Add positional encoding
        pose_with_pos = self.pos_encoder(pose_embedded)
        
        # [B, 1+S, M_dim]
        pose_features = self.transformer_encoder_pose(
            pose_with_pos, 
            mask=pose_mask, 
            src_key_padding_mask=pose_padding_mask
        )

        # --- 2. Audio Encoding ---
        # [B, S, A_dim] -> [B, S, M_dim]
        audio_embedded = self.audio_embed(audio_seq)
        
        # Add positional encoding
        audio_with_pos = self.pos_encoder(audio_embedded)
        
        # [B, S, M_dim]
        audio_features = self.transformer_encoder_audio(
            audio_with_pos, 
            mask=audio_mask, 
            src_key_padding_mask=audio_padding_mask
        )
        
        # --- 3. Cross-Attention Fusion ---
        # Query: Pose features (including [CLS] token) [B, 1+S, M_dim]
        # Key: Audio features [B, S, M_dim]
        # Value: Audio features [B, S, M_dim]
        fused_features, _ = self.cross_attention(
            query=pose_features, 
            key=audio_features, 
            value=audio_features,
            key_padding_mask=audio_padding_mask
        )
        
        # --- 4. Post-Fusion Block ---
        # Residual connection + LayerNorm
        fused_features = self.norm1(pose_features + fused_features)
        
        # FeedForward block + Residual connection + LayerNorm
        fused_ffn = self.ffn(fused_features)
        fused_features = self.norm2(fused_features + self.dropout_ffn(fused_ffn))

        # --- 5. Outputs ---
        # Embedding is the output of the [CLS] token (index 0)
        # [B, 1+S, M_dim] -> [B, M_dim]
        embedding = fused_features[:, 0]
        
        # Sequence features are the rest (from index 1 onwards)
        # [B, 1+S, M_dim] -> [B, S, M_dim]
        sequence_features = fused_features[:, 1:]
        
        # Project sequence features to prediction space
        # [B, S, M_dim] -> [B, S, P_dim]
        pred_pose = self.pose_head(sequence_features)
        
        # [B, S, M_dim] -> [B, S, A_dim]
        pred_audio = self.audio_head(sequence_features)

        return pred_pose, pred_audio, embedding

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class PoseAudioSequenceDataset(Dataset):
    """
    A dataset that returns shifted sequences for next-step prediction.
    
    Given a sequence of length L, it returns:
    - X (input): frames 0 to L-2
    - Y (target): frames 1 to L-1
    """
    def __init__(self, pose_sequences: list, audio_sequences: list):
        """
        Args:
            pose_sequences: A list of Tensors, where each is [seq_len, pose_dim]
            audio_sequences: A list of Tensors, where each is [seq_len, audio_dim]
        """
        # Ensure data is aligned
        assert len(pose_sequences) == len(audio_sequences)
        
        self.pose_sequences = []
        self.audio_sequences = []
        
        # Filter out sequences that are too short to train on (min length 2)
        for pose, audio in zip(pose_sequences, audio_sequences):
            length = min(pose.shape[0], audio.shape[0])
            pose = pose[:length]
            audio = audio[:length]
            assert pose.shape[0] == audio.shape[0]
            if pose.shape[0] > 1:
                self.pose_sequences.append(pose)
                self.audio_sequences.append(audio)

    def __len__(self):
        return len(self.pose_sequences)

    def __getitem__(self, idx):
        pose_seq = self.pose_sequences[idx]
        audio_seq = self.audio_sequences[idx]
        
        # X is the sequence from the beginning, dropping the last frame
        # Y is the sequence from the second frame, dropping the first
        # This creates the (input -> target) pairs for next-step prediction
        X_pose = pose_seq[:-1]
        X_audio = audio_seq[:-1]
        
        Y_pose = pose_seq[1:]
        Y_audio = audio_seq[1:]
        
        return X_pose, X_audio, Y_pose, Y_audio

def create_collate_fn(pad_value: float = 0.0):
    """
    Creates a collate function to batch and pad variable-length sequences
    and generate the necessary padding mask.
    """
    def collate_fn(batch):
        # batch is a list of tuples: [(X_p, X_a, Y_p, Y_a), ...]
        
        # Unzip the batch
        X_poses, X_audios, Y_poses, Y_audios = zip(*batch)
        
        # Pad all sequences in the batch to the length of the longest
        # pad_sequence expects a list of Tensors and returns a [B, T, *] Tensor
        X_poses_padded = pad_sequence(X_poses, batch_first=True, padding_value=pad_value)
        X_audios_padded = pad_sequence(X_audios, batch_first=True, padding_value=pad_value)
        Y_poses_padded = pad_sequence(Y_poses, batch_first=True, padding_value=pad_value)
        Y_audios_padded = pad_sequence(Y_audios, batch_first=True, padding_value=pad_value)
        
        # === Create the padding mask ===
        # This mask indicates which elements are padding (True) and which are real data (False)
        max_len = X_poses_padded.shape[1]
        
        # Get the original (unpadded) lengths of each sequence
        lengths = [len(x) for x in X_poses]
        lengths_tensor = torch.tensor(lengths)
        
        # Create a [B, max_len] mask
        # torch.arange(max_len)[None, :] -> [1, max_len]
        # lengths_tensor[:, None]        -> [B, 1]
        # The comparison broadcasts to [B, max_len]
        # Result: True where index >= length, False otherwise
        padding_mask = torch.arange(max_len)[None, :] >= lengths_tensor[:, None]
        
        return X_poses_padded, X_audios_padded, Y_poses_padded, Y_audios_padded, padding_mask

    return collate_fn



# === 1. Hyperparameters ===
POSE_DIM = 150       # 50 keypoints * 3D
AUDIO_DIM = 128      # 128 MFCC features
MODEL_DIM = 512
NHEAD = 8
NUM_LAYERS = 6
DIM_FEEDFORWARD = 2048
MAX_LEN = 1000
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 2. Create Dummy Data ===
# In a real scenario, you would load your data here.
# We create a list of 100 sequences with variable lengths.
dummy_pose_data = []
dummy_audio_data = []
for _ in range(100):
    seq_len = torch.randint(20, 100, (1,)).item() # Variable length
    dummy_pose_data.append(torch.randn(seq_len, POSE_DIM))
    dummy_audio_data.append(torch.randn(seq_len, AUDIO_DIM))

# === 3. Setup DataLoader ===
dataset = PoseAudioSequenceDataset(dummy_pose_data, dummy_audio_data)
custom_collate_fn = create_collate_fn(pad_value=0.0)
data_loader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    collate_fn=custom_collate_fn
)

# === 4. Initialize Model and Optimizer ===
model = MultimodalPoseAudioTransformer(
    pose_dim=POSE_DIM,
    audio_dim=AUDIO_DIM,
    model_dim=MODEL_DIM,
    nhead=NHEAD,
    num_encoder_layers=NUM_LAYERS,
    dim_feedforward=DIM_FEEDFORWARD,
    max_seq_len=MAX_LEN
).to(DEVICE)

# Use a loss function that doesn't reduce, so we can apply our own mask
pose_loss_fn = nn.MSELoss(reduction='none')
audio_loss_fn = nn.MSELoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# === 5. Training Step ===
model.train()
for batch in data_loader:
    # Get data and move to device
    X_pose, X_audio, Y_pose, Y_audio, padding_mask = [d.to(DEVICE) for d in batch]
    
    batch_size, seq_len = X_pose.shape[0], X_pose.shape[1]

    # --- Create Masks ---
    # 1. Causal (Look-ahead) Masks
    #    Prevents the model from seeing future tokens during self-attention
    #    Needs +1 for the [CLS] token on the pose side
    pose_causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len + 1).to(DEVICE)
    audio_causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(DEVICE)

    # 2. Padding Masks
    #    We need to adapt the pose padding mask for the [CLS] token
    #    Add a 'False' (not padded) column at the beginning for the [CLS] token
    cls_padding = torch.zeros(batch_size, 1, dtype=torch.bool, device=DEVICE)
    pose_padding_mask = torch.cat([cls_padding, padding_mask], dim=1)
    audio_padding_mask = padding_mask # Unmodified

    # --- Forward Pass ---
    pred_pose, pred_audio, embedding = model(
        pose_seq=X_pose,
        audio_seq=X_audio,
        pose_mask=pose_causal_mask,
        audio_mask=audio_causal_mask,
        pose_padding_mask=pose_padding_mask,
        audio_padding_mask=audio_padding_mask
    )
    
    # --- Compute Masked Loss ---
    # We must not compute loss on the padded (fake) tokens.
    # We use the original 'padding_mask' (shape [B, S])
    # ~padding_mask is 'True' for real data, 'False' for padding
    loss_mask = ~padding_mask
    
    # 1. Calculate unreduced pose loss: [B, S, P_dim]
    loss_pose_unreduced = pose_loss_fn(pred_pose, Y_pose)
    
    # 2. Mask the loss: [B, S, P_dim] * [B, S, 1] -> [B, S, P_dim]
    #    This zeros out the loss for all padded time steps
    loss_pose_masked = loss_pose_unreduced * loss_mask.unsqueeze(-1)
    
    # 3. Compute the mean, but only over non-padded elements
    #    Sum all losses and divide by the total number of non-padded elements
    num_real_elements_pose = loss_mask.sum() * POSE_DIM
    loss_pose = loss_pose_masked.sum() / num_real_elements_pose

    # Repeat for audio
    loss_audio_unreduced = audio_loss_fn(pred_audio, Y_audio)
    loss_audio_masked = loss_audio_unreduced * loss_mask.unsqueeze(-1)
    num_real_elements_audio = loss_mask.sum() * AUDIO_DIM
    loss_audio = loss_audio_masked.sum() / num_real_elements_audio

    total_loss = loss_pose + loss_audio
    
    # --- Backward Pass ---
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    print(f"Loss: {total_loss.item():.4f}")
    
    # Once trained, the 'embedding' tensor is ready for your downstream classifier!
    break # Only run one batch for demo