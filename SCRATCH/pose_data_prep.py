import torch
import numpy as np
import glob
import os
import numpy as np
from scipy.interpolate import interp1d
import json

SELECTED_KEYPOINTS = np.concatenate(([0,1,2,3,4,5,6,7,8,9,10], [40, 42, 44, 45, 47, 49], [59, 60, 61, 62, 63, 64], [65, 66, 67, 68, 69, 70], [71, 74, 77, 80], [91,95,96,99,100,103,104,107,108,111],[112,116,117,120,121,124,125,128,129,132]), axis=0)


def normalize_pose(pose_seq, root_joint_idx, SCALE_IDX_1, SCALE_IDX_2):
    """
    Normalizes a pose sequence for translation and scale invariance.

    Args:
        pose_seq (torch.Tensor): A tensor of shape [seq_len, num_keypoints, dims]
        root_joint_idx (int): The index of the root joint (e.g., pelvis).
        neck_joint_idx (int): The index of the neck joint.

    Returns:
        torch.Tensor: The normalized pose sequence.
    """
    # 1. Centering
    # Get the root joint position for every frame
    # root_joint shape: [seq_len, 1, dims]
    root_joint = pose_seq[:, root_joint_idx:root_joint_idx + 1, :]
    
    # Subtract the root joint from all keypoints
    centered_pose = pose_seq - root_joint
    
    # 2. Scaling
    # Get the centered neck keypoint
    # neck_keypoint shape: [seq_len, 1, dims]
    dist_coords = centered_pose[:, SCALE_IDX_1:SCALE_IDX_1 + 1, :] - centered_pose[:, SCALE_IDX_2:SCALE_IDX_2 + 1, :]
    
    # Calculate the L2 norm (distance from root to neck)
    # scale_factor shape: [seq_len, 1, 1]
    scale_factor = torch.norm(dist_coords, p=2, dim=-1, keepdim=True)

    # Add a small epsilon for numerical stability
    epsilon = 1e-8
    
    # Divide all centered keypoints by the scale factor
    normalized_pose = centered_pose / (scale_factor + epsilon)
    
    return normalized_pose

def downsample_pose_sequence(original_embeddings, target_fps, original_fps, method = 'linear'):
    """
    Downsamples a pose sequence to the target frame rate.

    Args:
        pose_seq (torch.Tensor): A tensor of shape [seq_len, num_keypoints, dims]
        target_fps (int): The desired frame rate.
        original_fps (int): The original frame rate of the sequence.

    Returns:
        torch.Tensor: The downsampled pose sequence.
    """
    num_frames_orig = original_embeddings.shape[0]

    print(f"Original shape: {original_embeddings.shape}")
    print(f"Original FPS: {original_fps}, Target FPS: {target_fps}")

    # --- 3. CREATE TIME VECTORS ---
    # Calculate total duration in seconds
    duration_sec = num_frames_orig / original_fps

    # Create the original time vector (timestamps for each frame)
    # Example: [0.0, 0.016, 0.033, ...] for 60fps
    t_original = np.linspace(0, duration_sec, num=num_frames_orig, endpoint=False)

    # Calculate the new number of frames
    num_frames_new = int(duration_sec * target_fps)

    # Create the new time vector (timestamps where we want to sample)
    # Example: [0.0, 0.041, 0.083, ...] for 24fps
    t_new = np.linspace(0, duration_sec, num=num_frames_new, endpoint=False)

    print(f"Original frames: {num_frames_orig}, New frames: {num_frames_new}")

    # --- 4. PERFORM INTERPOLATION ---
    print("Interpolating...")
    # We must interpolate along the time axis (axis=0)
    # 'kind='linear'' is a good default. You can also try 'cubic' for smoother
    # results, but it's computationally more expensive.
    f = interp1d(t_original, original_embeddings, axis=0, kind=method, fill_value="extrapolate")

    # Generate the new embeddings by sampling at the new timestamps
    resampled_embeddings = f(t_new)

    # --- 5. SAVE THE NEW DATA ---
    return torch.from_numpy(resampled_embeddings).float()

# --- Example Usage ---
# Assume you are using the COCO keypoint format (17 joints)
# Pelvis = (L_Hip[11] + R_Hip[12]) / 2
# Neck = (L_Shoulder[5] + R_Shoulder[6]) / 2
#
# If your keypoint detector doesn't provide these, you must add them.
# Let's assume you've pre-calculated and added them at:
# Index 17: Pelvis
# Index 18: Neck

ROOT_IDX = 1 # nose
SCALE_IDX_1 = 6 # right shoulder
SCALE_IDX_2 = 7 # left shoulder
with open("video_fps.json", "r") as f:
    fps_dict = json.load(f)
# chapter_pose_dict = {}
for file in glob.glob('../data/selected_poses/*.npy'):
    print(f"Processing file: {file}")
    seq = torch.from_numpy(np.load(file)).float()  # Example loading sequence
    # get x,y coords only
    seq_xy = seq[:, :, :2]
    probs = seq[:, :, 2:]

    normalized_data = normalize_pose(seq_xy, ROOT_IDX, SCALE_IDX_1, SCALE_IDX_2).numpy()
    # combine back with probs if needed
    normalized_data_with_probs = np.concatenate([normalized_data, probs.numpy()], axis=-1)[:, SELECTED_KEYPOINTS, :]

    # collapse to be [seq_len, num_keypoints*3]
    normalized_data_with_probs = torch.from_numpy(normalized_data_with_probs).float()
    normalized_data_with_probs = normalized_data_with_probs.reshape(normalized_data_with_probs.shape[0], -1)

    chapter = file.split("-")[-1].replace(".npy", "")
    original_fps = fps_dict[chapter]
    print(f"Original FPS for {file}: {original_fps}")
    normalized_data_with_probs = downsample_pose_sequence(normalized_data_with_probs, target_fps=24, original_fps=original_fps)
    print(f"Normalized data shape: {normalized_data_with_probs.shape}")
    # chapter_pose_dict[os.path.basename(file)] = normalized_data_with_probs

    # save data
    np.save(f"../data/normalized_poses/{os.path.basename(file)}", normalized_data_with_probs.numpy())

for file in glob.glob('../data/frame_labels/*.npy'):
    print(f"Processing file: {file}")
    seq = np.load(file)
    chapter = file.split("-")[-1].replace(".npy", "")
    original_fps = fps_dict[chapter]
    print(f"Original FPS for {file}: {original_fps}")
    downsampled_labels = downsample_pose_sequence(torch.from_numpy(seq).int(), target_fps=24, original_fps=original_fps, method='nearest').int()
    print(f"Downsampled labels shape: {downsampled_labels.shape}")

    # save data
    np.save(f"../data/downsampled_labels/{os.path.basename(file)}", downsampled_labels.numpy())
# sort by chapter, then concat
# chapters = sorted(chapter_pose_dict.keys())
# join on new external axis
# all_poses = np.stack([chapter_pose_dict[chapter].numpy() for chapter in chapters], axis=0)
# np.save("../data/all_normalized_poses.npy", all_poses.numpy())
# print(f"All poses concatenated shape: {all_poses.shape}")


# chapter_audio_dict = {}
# for file in glob.glob('../data/audio_features/*.npy'):
#     print(f"Processing file: {file}")
#     seq = np.load(file)

#     chapter_audio_dict[os.path.basename(file)] = seq

#     print(f"Audio data shape: {seq.shape}")
# # sort by chapter, then concat
# chapters = sorted(chapter_audio_dict.keys())
# all_audio = np.stack([chapter_audio_dict[chap] for chap in chapters], axis=0)
# np.save("../data/all_audio_features.npy", all_audio)
# print(f"All audio concatenated shape: {all_audio.shape}")