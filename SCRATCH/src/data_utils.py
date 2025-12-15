import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from src.config import NORMALIZED_POSES_DIR, AUDIO_FEATURES_DIR

class RawExtractionDataset(Dataset):
    """
    Dataset used ONLY for the initial pass to extract embeddings from the base model.
    It accepts pre-calculated window_indices derived from the CSV.
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
    
    # Create mask for transformer padding
    lengths = [len(x) for x in X_p]
    max_len = X_p_pad.shape[1]
    mask = torch.arange(max_len)[None, :] >= torch.tensor(lengths)[:, None]
    
    return X_p_pad, X_a_pad, y_tensor, mask

def load_raw_files(chapters, type=None):
    """Loads the .npy files for the given chapters."""
    p_data, a_data = [], []
    for ch in chapters:
        if type == 'pose' or type is None:
            p_data.append(np.load(f"{NORMALIZED_POSES_DIR}/poses-{ch}.npy"))
        if type == 'audio' or type is None:
            a_data.append(np.load(f"{AUDIO_FEATURES_DIR}/audio-{ch}.npy"))
    if type == 'pose':
        return p_data
    elif type == 'audio':
        return a_data
    else:
        return p_data, a_data

def parse_csv_windows(segments_df, chapters, categories_to_include, fps):
    """
    Parses the CSV to create (sequence_index, start_frame) tuples for given categories.
    """
    window_indices = []
    labels = []
    
    # Create a mapping from chapter name to its index in the loaded data list
    chapter_to_seq_idx = {ch: i for i, ch in enumerate(chapters)}
    
    # Filter dataframe for the relevant chapters and categories
    df_filtered = segments_df[segments_df['pair_speaker'].isin(chapters) & segments_df['category'].isin(categories_to_include)]

    for _, row in df_filtered.iterrows():
        chapter = row['pair_speaker']
        seq_idx = chapter_to_seq_idx[chapter]
        start_frame = int(row['from_ts'] * fps)
        label = int(row['is_present'])
        
        window_indices.append((seq_idx, start_frame))
        labels.append(label)
            
    return window_indices, labels