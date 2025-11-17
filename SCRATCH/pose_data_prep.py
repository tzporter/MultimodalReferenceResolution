import torch
import numpy as np
import glob
import os

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
chapter_pose_dict = {}
for file in glob.glob('../data/selected_poses/*.npy'):
    print(f"Processing file: {file}")
    seq = torch.from_numpy(np.load(file)).float()  # Example loading sequence
    # get x,y coords only
    seq_xy = seq[:, :, :2]
    probs = seq[:, :, 2:]

    normalized_data = normalize_pose(seq_xy, ROOT_IDX, SCALE_IDX_1, SCALE_IDX_2)

    # combine back with probs if needed
    normalized_data_with_probs = torch.cat([normalized_data, probs], dim=-1)

    # collapse to be [seq_len, num_keypoints*3]
    normalized_data_with_probs = normalized_data_with_probs.view(normalized_data_with_probs.shape[0], -1)

    print(f"Normalized data shape: {normalized_data_with_probs.shape}")
    chapter_pose_dict[os.path.basename(file)] = normalized_data_with_probs

    # save data
    np.save(f"../data/normalized_poses/{os.path.basename(file)}", normalized_data_with_probs.numpy())
# sort by chapter, then concat
chapters = sorted(chapter_pose_dict.keys())
all_poses = torch.cat([chapter_pose_dict[chap] for chap in chapters], dim=0)
print(f"All poses concatenated shape: {all_poses.shape}")