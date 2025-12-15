import torch

# ==========================================
# Hyperparameters & Setup
# ==========================================

# --- Model Dimensions ---
POSE_DIM = 159
AUDIO_DIM = 128
MODEL_DIM = 256
NHEAD = 8
NUM_LAYERS = 6
DIM_FEEDFORWARD = 1024

# --- Data & Windowing ---
WINDOW_SIZE = 72
STEP_SIZE = WINDOW_SIZE // 4  # Used only if we needed to generate sliding windows
FPS = 24

# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Classification & Reduction ---
NUM_CLASSES = 2
REDUCED_DIMENSION = 256  # PCA output dimension
FINETUNE_BATCH_SIZE = 32
FINETUNE_EPOCHS = 100
FINETUNE_LR = 0.0005

# --- Paths ---
# Note: These paths are relative to the project root (where run_experiment.py is)
BEST_CHECKPOINT_PATH = "checkpoints/epoch_34_valloss_1.0583.pth"
SPEAKER_SEGMENTS_PATH = "../data/speaker_segments.csv"
CHAPTERS_JSON_PATH = "train_test_chapters.json"
NORMALIZED_POSES_DIR = "../data/normalized_poses"
AUDIO_FEATURES_DIR = "../data/audio_features"