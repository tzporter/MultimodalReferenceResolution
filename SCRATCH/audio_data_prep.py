import cv2
import librosa
import torch
import numpy as np
import glob
import os

def extract_audio_features(video_path: str, audio_dim: int, video_fps: float) -> torch.Tensor:
    """
    Extracts temporally aligned audio features (MFCCs) from a video file.

    Args:
        video_path: Path to the .mp4 file.
        audio_dim: The desired feature dimension (n_mfcc).

    Returns:
        A torch.Tensor of shape [seq_len, audio_dim].
    """
    
    # 1. Get Video FPS using OpenCV
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file {video_path}")

        # video_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Extracted FPS from video: {video_fps}")
        num_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total video frames: {num_video_frames}")
        cap.release()
        
        if video_fps <= 0:
            print(f"Warning: Could not get valid FPS for {video_path}. Defaulting to 30.0")
            video_fps = 30.0
            
    except Exception as e:
        print(f"Error reading video properties with OpenCV: {e}")
        return None

    # 2. Load Audio Waveform using Librosa
    # try:
        # Load at original sample rate
    waveform, sample_rate = librosa.load(video_path, sr=None)
    # except Exception as e:
        # print(f"Error loading audio with Librosa (check ffmpeg): {e}")
        # return None

    # 3. Calculate Alignment (hop_length)
    # This is the number of audio samples per video frame
    hop_length = int(sample_rate / video_fps)
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Calculated hop_length: {hop_length} samples")


    # 4. Extract MFCCs
    # n_mfcc is our feature dimension (audio_dim)
    # We use standard parameters for n_fft
    mfccs = librosa.feature.mfcc(
        y=waveform,
        sr=sample_rate,
        n_mfcc=audio_dim,
        hop_length=hop_length,
        n_fft=int(hop_length * 2) # A common heuristic
    )
    
    # 5. Transpose and Convert to Tensor
    # Librosa returns [audio_dim, seq_len], we want [seq_len, audio_dim]
    mfccs_transposed = mfccs.T
    
    # --- 6. Final Alignment (Crucial!) ---
    # The number of audio features might be off by 1-2 frames from 
    # the number of video frames due to windowing.
    # We truncate to the minimum of the two.
    num_audio_frames = mfccs_transposed.shape[0]
    
    # You will need to do this for your pose data as well
    # E.g., num_video_frames = X_pose.shape[0]
    
    final_len = min(num_video_frames, num_audio_frames)
    
    final_audio_features = mfccs_transposed[:final_len]
    
    # Convert to a PyTorch tensor
    X_audio = torch.tensor(final_audio_features, dtype=torch.float32)

    print(f"Video FPS: {video_fps:.2f}")
    print(f"Audio Sample Rate: {sample_rate} Hz")
    print(f"Hop Length: {hop_length} samples")
    print(f"Original video frames: {num_video_frames}, Audio features: {num_audio_frames}")
    print(f"Final aligned shape: {X_audio.shape}")
    
    return X_audio

# --- How to use it ---
# VIDEO_FILE = "../../ASD_video/videos/input/ch07_speakerview.mp4"
AUDIO_DIM = 128 # This must match your model's audio_dim

# X_audio is the tensor for your data loader
for video_file in glob.glob('../videos/input/*.mp4'):
    print(f"Processing video file: {video_file}")
    X_audio = extract_audio_features(video_file, AUDIO_DIM, video_fps=24.0)
    chapter = os.path.basename(video_file).replace('_speakerview.mp4', '')
    if X_audio is not None:
        # Save or use X_audio as needed
        np.save(f"./SCRATCH/audio_features/audio-{chapter}", X_audio.numpy())
    else:
        print(f"Failed to extract audio features for {video_file}")

# Now, you must ALSO trim your X_pose data to match this length!
# e.g., X_pose = X_pose[:X_audio.shape[0]]