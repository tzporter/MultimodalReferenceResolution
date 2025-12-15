import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from src.models import MultimodalPoseAudioTransformer
from src.data_utils import extraction_collate_fn
from src.config import (
    DEVICE, POSE_DIM, AUDIO_DIM, MODEL_DIM, NHEAD, NUM_LAYERS,
    DIM_FEEDFORWARD, WINDOW_SIZE, BEST_CHECKPOINT_PATH
)

def extract_features(dataset):
    """
    Performs a one-time feature extraction using the frozen base model.
    """
    print(f"Loading base model from {BEST_CHECKPOINT_PATH}...", flush=True)
    base_model = MultimodalPoseAudioTransformer(
        pose_dim=POSE_DIM, audio_dim=AUDIO_DIM, model_dim=MODEL_DIM,
        nhead=NHEAD, num_encoder_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
        max_seq_len=WINDOW_SIZE + 1
    ).to(DEVICE)
    
    checkpoint = torch.load(BEST_CHECKPOINT_PATH, map_location=DEVICE)
    base_model.load_state_dict(checkpoint['model_state_dict'])
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=extraction_collate_fn)
    embeddings, labels = [], []
    print(f"Extracting features for {len(dataset)} samples...")
    
    with torch.no_grad():
        for X_p, X_a, y, mask in loader:
            X_p, X_a, mask = X_p.to(DEVICE), X_a.to(DEVICE), mask.to(DEVICE)
            
            seq_len = X_p.shape[1]
            p_causal = nn.Transformer.generate_square_subsequent_mask(seq_len + 1).to(DEVICE)
            a_causal = nn.Transformer.generate_square_subsequent_mask(seq_len).to(DEVICE)
            cls_pad = torch.zeros(X_p.shape[0], 1, dtype=torch.bool, device=DEVICE)
            p_pad = torch.cat([cls_pad, mask], dim=1)
            
            _, _, embedding = base_model(
                pose_seq=X_p, audio_seq=X_a, pose_mask=p_causal, audio_mask=a_causal,
                pose_padding_mask=p_pad, audio_padding_mask=mask
            )
            embeddings.append(embedding.cpu().numpy())
            labels.append(y.numpy())
            
    return np.concatenate(embeddings), np.concatenate(labels)