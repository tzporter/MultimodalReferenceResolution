import torch
import torch.nn as nn
import math

# --- [PositionalEncoding class] ---
# (No changes needed)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe.transpose(0, 1)[:, :x.size(1)]
        return self.dropout(x)

# --- [MultimodalPoseAudioTransformer class] ---
# (No changes needed)
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
        super().__init__()
        self.model_dim = model_dim
        self.pose_embed = nn.Linear(pose_dim, model_dim)
        self.audio_embed = nn.Linear(audio_dim, model_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim))
        self.pos_encoder = PositionalEncoding(model_dim, dropout, max_seq_len)
        pose_encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=nhead, dim_feedforward=dim_feedforward, 
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder_pose = nn.TransformerEncoder(
            pose_encoder_layer, num_layers=num_encoder_layers
        )
        audio_encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=nhead, dim_feedforward=dim_feedforward, 
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder_audio = nn.TransformerEncoder(
            audio_encoder_layer, num_layers=num_encoder_layers
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=model_dim, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, model_dim)
        )
        self.dropout_ffn = nn.Dropout(dropout)
        self.pose_head = nn.Linear(model_dim, pose_dim)
        self.audio_head = nn.Linear(model_dim, audio_dim)

    def forward(self, 
                pose_seq: torch.Tensor, 
                audio_seq: torch.Tensor, 
                pose_mask: torch.Tensor, 
                audio_mask: torch.Tensor,
                pose_padding_mask: torch.Tensor,
                audio_padding_mask: torch.Tensor):
        batch_size = pose_seq.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        pose_embedded = self.pose_embed(pose_seq) 
        pose_embedded = torch.cat([cls_tokens, pose_embedded], dim=1)
        pose_with_pos = self.pos_encoder(pose_embedded)
        pose_features = self.transformer_encoder_pose(
            pose_with_pos, 
            mask=pose_mask, 
            src_key_padding_mask=pose_padding_mask
        )
        audio_embedded = self.audio_embed(audio_seq)
        audio_with_pos = self.pos_encoder(audio_embedded)
        audio_features = self.transformer_encoder_audio(
            audio_with_pos, 
            mask=audio_mask, 
            src_key_padding_mask=audio_padding_mask
        )
        fused_features, _ = self.cross_attention(
            query=pose_features, 
            key=audio_features, 
            value=audio_features,
            key_padding_mask=audio_padding_mask
        )
        fused_features = self.norm1(pose_features + fused_features)
        fused_ffn = self.ffn(fused_features)
        fused_features = self.norm2(fused_features + self.dropout_ffn(fused_ffn))
        embedding = fused_features[:, 0]
        sequence_features = fused_features[:, 1:]
        pred_pose = self.pose_head(sequence_features)
        pred_audio = self.audio_head(sequence_features)
        return pred_pose, pred_audio, embedding