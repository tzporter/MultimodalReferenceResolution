import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import sys
import random
sys.path.extend(['../'])
import torchaudio
from torchaudio import functional as F
import librosa
import glob
from utils.augmentation_utils import compose_random_augmentations
# from utils.utils_data import crop_scale

from tqdm import tqdm
mediapipe_flip_index = np.concatenate(([0,2,1,4,3,6,5], [28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48], [7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26, 27], ), axis=0) 

mmpose_flip_index = np.concatenate(([0,2,1,4,3,6,5],[17,18,19,20,21,22,23,24,25,26],[7,8,9,10,11,12,13,14,15,16]), axis=0) 
content_words_pos = ['ADJ', 'ADV', 'NOUN', 'PROPN', 'VERB', 'NUM']  # removed 'INTJ'
effect = ",".join(
    [
        "lowpass=frequency=300:poles=1",  # apply single-pole lowpass filter
        "atempo=0.8",  # reduce the speed
        "aecho=in_gain=0.8:out_gain=0.9:delays=200:decays=0.3|delays=400:decays=0.3"
        # Applying echo gives some dramatic feeling
    ],
)

# effect = [
#     ["lowpass", "-f", "300", "-p", "1"],  # apply single-pole lowpass filter
#     ["atempo", "0.8"],                   # reduce the speed
#     ["aecho", "0.8", "0.9", "200", "0.3", "400", "0.3"] # echo effect
# ]

# Define effects
effects = [
    ["lowpass", "-1", "300"],  # apply single-pole lowpass filter
   #  ["speed", "0.8"],  # reduce the speed
    # This only changes sample rate, so it is necessary to
    # add `rate` effect with original sample rate after this.
   #  ["rate", f"{sample_rate1}"],
    ["reverb", "-w"],  # Reverbration gives some dramatic feeling
]

configs = [
    {"format": "wav", "encoding": "ULAW", "bits_per_sample": 8},
    {"format": "gsm"},
   #  {"format": "vorbis", "compression": -1},
]
def load_audio_dict(pairs_speakers, audio_path, audio_sample_rate):
    audio_dict = {}
    print('Loading audio files...')
    for speaker in tqdm(pairs_speakers):
        pair = speaker.split('_')[0]
        speaker = speaker.split('_')[1]
        pair_speaker = f"{pair}_{speaker}"
        try:
            audio_path = audio_path.format(pair, speaker)
            input_audio, sample_rate = librosa.load(audio_path, sr=audio_sample_rate)
        except FileNotFoundError:
            audio_path = audio_path.format(pair, pair, speaker)
            input_audio, sample_rate = librosa.load(audio_path, sr=audio_sample_rate)
        audio_dict[pair_speaker] = {'audio': input_audio, 'sample_rate': sample_rate}
    return audio_dict
                
def custom_collate_fn(item):
    max_length = max([seq['skeleton']['orig'].shape[0] for seq in item])
    batch = [seq['skeleton']['orig'] for seq in item]
    padded_sequences = []
    masks = []
    for seq in batch:
        pad_size = max_length - seq.shape[0]
        padded_seq = torch.nn.functional.pad(seq, (0, 0, 0, 0, 0, pad_size), mode='constant', value=0)
        padded_sequences.append(padded_seq) 
        # Create a mask where 1 indicates actual data and 0 indicates padding
        mask = torch.cat((torch.ones(seq.shape[0]), torch.zeros(pad_size)))
        masks.append(mask)
    batch_masks = torch.stack(masks, dim=0)
    # replace the original sequences with the padded sequences in the item
    for i, seq in enumerate(batch):
        item[i]['skeleton']['orig'] = padded_sequences[i]
        item[i]['mask'] = batch_masks[i]
    return item 


def miror_poses(data_numpy):
   C,T,V,M = data_numpy.shape
   assert C == 3 # x,y,c
   data_numpy[0, :, :, :] = 1920 - data_numpy[0, :, :, :]
   return data_numpy

def process_poses(poses):
   # add one dimension to the pose at the end
   poses = np.expand_dims(poses, axis=-1)
   # original shape is T, V, C, M
   T, V, C, M = poses.shape
   poses = np.transpose(poses, (2, 0, 1, 3))

   mirrod_poses = miror_poses(poses.copy())
   return poses, mirrod_poses
       
def load_keypoints_dict(all_pair_speakers, selected_keypoints):
   # read all poses in f'data/final_poses/poses_{pair}_synced_pp{speaker}.npy'
   processed_keypoints_dict = {}
   mirrored_keypoints_dict = {}
   # Modify this path to point to your directory of .npy files
   for file_path in tqdm(glob.glob('./data/selected_poses/*.npy'), desc='Loading keypoints...'):
      pair_speaker = file_path.split('-')[-1].replace('.npy', '')
      if pair_speaker not in all_pair_speakers:
            continue
      try:
        # select only keypoints in selected_keypoints
         selected_poses = np.load(file_path)[:, selected_keypoints, :]
         processed_keypoints_dict[pair_speaker], mirrored_keypoints_dict[pair_speaker] = process_poses(selected_poses)
      except Exception as e:
         print(e)
         print('Error in loading the keypoints for the pair:', pair, speaker)
         continue
   return processed_keypoints_dict, mirrored_keypoints_dict


class CABBFeeder(Dataset):
    def __init__(
            self,
            random_choose=True,
            random_shift=True,
            random_move=True,
            window_size=30,
            normalization=True,
            debug=False,
            use_mmap=True,
            random_mirror=True,
            random_mirror_p=0.5,
            is_vector=False,
            fold=0,
            sample_rate1=16000,
            debug_audio=False,
            data_path='./dialog_utils/data/gestures_info_with_text.csv',
            poses_path='./data/',
            modalities=["skeleton", "speech"],
            audio_path='data/{}_synced_pp{}.wav',
            n_views=2,
            apply_skeleton_augmentations=True,
            skeleton_augmentations=None,
            random_scale=True,
            fps=24,
            speech_buffer=0.5,
            gesture_buffer=0.25,
            apply_speech_augmentations=False,
            global_normalization=False,
            phase='train',
            task='pre-training',
            use_only_small_dataset=False,
            filter_text=True,
            crop_scale=False,
            skeleton_backbone='jointsformer',
            eval_on_small_dataset=True,
            train_data=True,
            selected_keypoints = np.concatenate(([0,5,6,7,8,9,10], [91,95,96,99,100,103,104,107,108,111],[112,116,117,120,121,124,125,128,129,132]), axis=0)
    ):
        """ 
        :param poses_path: path to poses 
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move: 
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param data_path: path to data
        :param modalities: list of modalities
        :param n_views: number of views (1 or 2)
        :param apply_augmentations: whether to apply augmentations to view1, view2 is augmented by default
        """
        self.data_path = data_path
        self.poses_path = poses_path
        self.audio_path = audio_path
        self.selected_keypoints = selected_keypoints
        self.modalities = modalities
        self.phase = phase
        self.apply_speech_augmentations = apply_speech_augmentations
        self.global_normalization = global_normalization
        self.apply_skeleton_augmentations=apply_skeleton_augmentations
        self.skeleton_augmentations=skeleton_augmentations
        self.random_scale = random_scale
        self.task = task
        self.use_only_small_dataset = use_only_small_dataset
        self.skeleton_backbone = skeleton_backbone
        self.eval_on_small_dataset = eval_on_small_dataset
        self.train_data = train_data

        if apply_skeleton_augmentations:
            assert self.skeleton_augmentations is not None, "Augmentations are not provided"
        self.n_views=n_views
        

        self.debug_audio = debug_audio
        self.debug = debug
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.random_mirror = random_mirror
        self.random_mirror_p = random_mirror_p
        self.audio_sample_rate = sample_rate1
        self.fps = fps
        self.speech_buffer = speech_buffer
        self.gesture_buffer = gesture_buffer

        self.filter_text = filter_text
        self.crop_scale = crop_scale

        self.data = None
        self.mirrored_poses = None
        self.pairs_speakers = None
        self.poses = None
        self.load_data()
        self.is_vector = is_vector
        if 'speech' not in self.modalities:
            self.audio_dict = None

        if self.global_normalization:
            self.get_mean_map()
        self.audio_sample_per_frame = int(self.audio_sample_rate/self.fps)
        # self.data = self.data[-640:]

        # SAMPLE_RIR = download_asset("tutorial-assets/Lab41-SRI-VOiCES-rm1-impulse-mc01-stu-clo-8000hz.wav")
        # SAMPLE_NOISE = download_asset("tutorial-assets/Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo-8000hz.wav")
        # rir_raw, rir_sample_rate = torchaudio.load(SAMPLE_RIR)
        # self.rir_sample_rate = rir_sample_rate
        # self.noise, noise_sample_rate = torchaudio.load(SAMPLE_NOISE)
        # # target sample rate is 16000
        # self.noise = F.resample(self.noise, orig_freq=noise_sample_rate, new_freq=sample_rate1)
        # rir = rir_raw[:, int(rir_sample_rate * 1.01) : int(rir_sample_rate * 1.3)]
        # self.rir = rir / torch.linalg.vector_norm(rir, ord=2)

    def prepared_segmentation_sequences(self):
        max_sequnce_length = self.window_size
        frame_offset = self.window_size//2
        # iterate over the poses
        gesture_sequences_data = []
        # convert start_frames and end_frames to int
        self.data['start_frames'] = self.data['start_frames'].astype(int)
        self.data['end_frames'] = self.data['end_frames'].astype(int)
        # filter our 0 length gestures
        self.data = self.data[self.data['end_frames'] >= self.data['start_frames']]

        for pair_speaker in self.pairs_speakers:
            # select the rows of the pair_speaker
            data = self.data[self.data['pair_speaker'] == pair_speaker]
            # reset the index of the data
            data.reset_index(drop=True, inplace=True)
            total_frames = self.poses[pair_speaker].shape[1]
            frame_gesture_list = [0] * total_frames
            # Set the frames with gestures to 1
            start_frames = data['start_frames'].values
            end_frames = data['end_frames'].values
            # Iterate over the start and end frames to set the frames with gestures to 1
            for start_frame, end_frame in zip(start_frames, end_frames):
                for i in range(start_frame, end_frame + 1):
                    frame_gesture_list[i] = 1

            # TODO check how to incorporate the frame offset
            start_frame = 0
            end_frame = self.poses[pair_speaker].shape[1] - max_sequnce_length
            for f in range(start_frame, end_frame, frame_offset):
                labels = frame_gesture_list[f:f + max_sequnce_length]
                if sum(labels) == 0:
                    is_gesture = 'non-gesture'
                else:
                    is_gesture = 'gesture'

                gesture_instance = {
                    'pair_speaker': pair_speaker,
                    'start_frames': f,
                    'end_frames': f + max_sequnce_length,
                    'labels': frame_gesture_list[f:f + max_sequnce_length],
                    'speaker': pair_speaker.split('_')[1],
                    'pair': pair_speaker.split('_')[0],
                    'is_gesture': is_gesture,
                    'contain_text': False,
                    'words': 'geen speech',
                    'segmentation method': 'manual',
                    'from_ts': f / self.fps,
                    'to_ts': (f + max_sequnce_length) / self.fps,
                    'probability': 1.0,
                    'referent': 'None'
                }  
                gesture_sequences_data.append(gesture_instance)
        self.data = pd.DataFrame(gesture_sequences_data)              

    def load_skeletal_data(self):
        # data: N C V T M
        if self.data_path.endswith('.csv'):
            self.data = pd.read_csv(self.data_path)
            self.data['start_frames'] = self.data['from_ts'] * self.fps
            self.data['end_frames'] = self.data['to_ts'] * self.fps
        elif self.data_path.endswith('.pkl'):
            self.data = pd.read_pickle(self.data_path)
        self.data = self.data.sample(frac=1, random_state=42).reset_index(drop=True)
        # filter out 0 length gestures
        # self.data = self.data[self.data['end_frames'].astype(int)-self.data['start_frames'].astype(int) > 100]

        if self.eval_on_small_dataset and not self.use_only_small_dataset and self.task != 'segmentation':
            if self.train_data: 
                if self.data_path.endswith('.csv'):
                    self.data = self.data[self.data['segmentation method'] != 'manual']
                    # self.data = self.data[self.data['segmentation method'] == 'skeleton&speech model']
                else:
                    self.data = self.data[self.data['segmentation method'] != 'manual']
            else: 
                self.data = self.data[self.data['segmentation method'] == 'manual']
        elif self.task == 'segmentation' or self.use_only_small_dataset:
            self.data = self.data[self.data['segmentation method'] == 'manual']
        if self.filter_text:
            self.data = self.data[self.data["words"] != "0"]
            # Select rows where the number of words in 'words' and 'pos' columns are equal
            self.data['word_count'] = self.data['words'].apply(lambda x: len(str(x).split()))
            # self.data['pos_count'] = self.data['pos'].apply(lambda x: len(str(x).split()))
            # self.data = self.data[self.data['word_count'] == self.data['pos_count']]
            # Filter rows where at least one POS tag is in content_words_pos
            # self.data['has_content'] = self.data['pos'].apply(
                # lambda x: any(pos in content_words_pos for pos in str(x).split())
            # )
            # self.data = self.data[self.data['has_content']]

            # Keep only content words in the 'words' column
            # def filter_content_words(words, pos):
                # return ' '.join([word for word, p in zip(words.split(), pos.split()) if p in content_words_pos])

            # self.data['words'] = self.data.apply(lambda row: filter_content_words(row['words'], row['pos']), axis=1)

            # Remove rows where all words have been filtered out
            # self.data = self.data[self.data['words'] != '']

            # Drop the temporary column used for filtering
            # self.data = self.data.drop(columns=['has_content', 'word_count', 'pos_count'])
            
            # reindex the data
            self.data = self.data.reset_index(drop=True)
        # convert from_ts and to_ts to frames
        self.data = self.data.reset_index(drop=True)
        if self.debug:
            # select rows of pair04
            self.data = self.data[self.data['pair_speaker'] == 'pair04_A']
        self.pairs_speakers = self.data['pair_speaker'].unique()
        self.poses = {}
        self.poses, self.mirrored_poses = load_keypoints_dict(self.pairs_speakers, self.selected_keypoints)
        # select only poses of the pairs in the data
        self.poses = {pair_speaker: self.poses[pair_speaker] for pair_speaker in self.pairs_speakers}
        self.mirrored_poses = {pair_speaker: self.mirrored_poses[pair_speaker] for pair_speaker in self.pairs_speakers}
        if self.task == 'segmentation':
            self.prepared_segmentation_sequences()
      

    def load_speech_data(self):
        if not self.debug_audio:
            self.audio_dict = load_audio_dict(self.pairs_speakers, self.audio_path, self.audio_sample_rate)

            

    def load_data(self):
        if "skeleton" in self.modalities:
            self.load_skeletal_data()
        if "speech" in self.modalities:
            self.load_speech_data()

    # TODO: define augmentations from HAR project. Should the strength of augmentations be the same?
    def augment_skeleton(self, data_numpy, config_dict):
        augmentations = transforms.Compose(compose_random_augmentations("skeleton", config_dict))
        return augmentations(data_numpy)

    def apply_codec(self, waveform, orig_sample_rate, **kwargs):
        if orig_sample_rate != 8000:
            waveform = F.resample(waveform, orig_sample_rate, 8000)
            sample_rate = 8000
        augmented = F.apply_codec(waveform, sample_rate, **kwargs)
        # effector = torchaudio.io.AudioEffector(effect=effect)
        # augmented_effector = effector.apply(waveform, sample_rate)
        # resample to original sample rate
        augmented = F.resample(augmented, sample_rate, orig_sample_rate)
        return augmented

    def augment_audio(self, audio, augemntation_apply=True):
        if not augemntation_apply:
            return audio
        # apply effects
        lengths = audio.shape[0]
        audio = torch.from_numpy(audio).float().unsqueeze(0)
        # apply effects with 50% probability
        coin_toss = random.random()
        if coin_toss < 0.33:
            audio, _ = torchaudio.sox_effects.apply_effects_tensor(audio, self.audio_sample_rate, effects)
            # choose randomly one augmented speech from the two augmented speech
            idx = random.randint(0, audio.shape[0] - 1)
            audio = audio[idx].unsqueeze(0)
            augemntation_apply = False
        elif coin_toss < 0.66:
            if self.noise.shape[1] < audio.shape[1]:
                noise = self.noise.repeat(1, 2)[:,:audio.shape[1]]
            else:
                noise = self.noise[:, : audio.shape[1]]
            snr_dbs = torch.tensor([20, 10, 3])
            audio = F.add_noise(audio, noise, snr_dbs)
            # choose randomly one noisy speech
            idx = random.randint(0, audio.shape[0] - 1)
            audio = audio[idx].unsqueeze(0)
            augemntation_apply = False
        else:
            waveforms = []
            for param in configs:
                augmented = self.apply_codec(audio, self.audio_sample_rate, **param)
                waveforms.append(augmented)
            # choose randomly one codec
            idx = random.randint(0, len(waveforms) - 1)
            audio = waveforms[idx]
            augemntation_apply = False
            if audio.shape[1] > lengths: # TODO: check the validity of this operation: if the augmented speech is longer than the original speech, truncate it
                audio = audio[:, :lengths]
        audio = audio.squeeze(0).numpy()
        return audio

    def get_mean_map(self):
        data = self.get_all_poses()
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def get_all_poses(self):
        # reset index of the data
        all_poses = np.zeros((len(self.data), 3, 30, 27, 1))
        for index, row in tqdm(self.data.iterrows(), total=self.data.shape[0], desc='Loading poses...'):
            start_frame = int(row['start_frames'])  # - round(self.gesture_buffer * self.fps)
            end_frame = int(row['end_frames'])  # + round(self.gesture_buffer * self.fps)
            middle_frame = (start_frame + end_frame) // 2
            # take 15 frames before and after the middle frame
            start_frame = middle_frame - 15
            end_frame = start_frame + 30
            pair_speaker = row['pair_speaker']
            all_poses[index] = self.poses[pair_speaker][:, start_frame:end_frame, :, :]
        return all_poses

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        if 'labels' in self.data.columns and self.task == 'segmentation':
            item = {
                "label": np.array(self.data.iloc[index]['labels'])
            }
        else:
            item = {
                "label": 0
            }
        row = self.data.iloc[index]
        # check if embeddings are present
        # try:
        if 'semantic' in self.modalities:
            # check if row['words'] is a string and not None
            if self.filter_text:
                item['utterance'] = row['words']
            elif row['words'] and isinstance(row['words'], str):
                speech = row['words'].split()
                # pos_seq = row['pos'].split()
                # assert len(speech) == len(pos_seq)
                content_words = ' '.join(speech)
                item['utterance'] = content_words
            else:
                item['utterance'] = 'geen speech'
       
        start_frame = int(row['start_frames'])
        end_frame = int(row['end_frames'])
        if self.random_choose and self.phase == 'train':
            # random choose a portion
            if end_frame - start_frame > self.window_size:
                start_frame = np.random.randint(start_frame, end_frame - self.window_size + 1)
            else:
                middle_frame = (start_frame + end_frame) // 2
                # take 15 frames before and after the middle frame 
                start_frame = middle_frame - round(self.window_size / 2)
            end_frame = start_frame + self.window_size
        speech_start_frame = start_frame - round(self.speech_buffer * self.fps)
        # speech_end_frame = speech_start_frame + round(self.speech_buffer * self.fps)
        pair_speaker = row['pair_speaker']
        number_speech_frames = int(2 * self.audio_sample_rate)
        # add speaker and pair to the item
        speaker_dict = {'A': 0, 'B': 1}
        # item['speaker_ID'] = speaker_dict[pair_speaker.split('_')[1]]
        # item['pair_ID'] = int(pair_speaker.split('_')[0].split('pair')[-1])
        # add frame_ID to the item
        frame_IDs = np.arange(start_frame, end_frame)
        assert len(frame_IDs) >= 30, (start_frame, end_frame)
        item['frame_IDs'] = frame_IDs
        if "skeleton" in self.modalities:
            # select either the original or the mirrored poses
            if self.random_mirror and random.random() > self.random_mirror_p:
                assert np.array(self.mirrored_poses[pair_speaker]).shape[1] > end_frame, (pair_speaker, self.mirrored_poses[pair_speaker].shape, end_frame)
                skeleton_data = self.mirrored_poses[pair_speaker][:, start_frame:end_frame, :, :]  
            else:
                assert np.array(self.poses[pair_speaker]).shape[1] > end_frame, (pair_speaker, self.poses[pair_speaker].shape, end_frame)
                skeleton_data = self.poses[pair_speaker][:, start_frame:end_frame, :, :]          
            item["skeleton"] = {}
            # skeleton_data = self.data[index]
            skeleton_data_numpy = np.array(skeleton_data)
            if self.crop_scale:
                skeleton_data_numpy = np.transpose(skeleton_data_numpy, (3, 1, 2, 0))
                # skeleton_data_numpy = crop_scale(skeleton_data_numpy)
                skeleton_data_numpy = np.transpose(skeleton_data_numpy, (3, 1, 2, 0))
            elif self.normalization:
                assert skeleton_data_numpy.shape[0] == 3
                if self.global_normalization:
                    skeleton_data_numpy = (skeleton_data_numpy - self.mean_map) / self.std_map
                elif self.is_vector:
                    skeleton_data_numpy[0, :, 0, :] = skeleton_data_numpy[0, :, 0, :] \
                        - skeleton_data_numpy[0, :, 0, 0].mean(axis=0)
                    skeleton_data_numpy[1, :, 0, :] = skeleton_data_numpy[1, :, 0, :] \
                        - skeleton_data_numpy[1, :, 0, 0].mean(axis=0)
                else:
                    skeleton_data_numpy[0, :, :, :] = skeleton_data_numpy[0, :, :, :] \
                        - skeleton_data_numpy[0, :, 0, 0].mean(axis=0)
                    skeleton_data_numpy[1, :, :, :] = skeleton_data_numpy[1, :, :, :] \
                        - skeleton_data_numpy[1, :, 0, 0].mean(axis=0)
            item["skeleton"]["orig"] = skeleton_data_numpy
            # assert skeleton_data_numpy.shape[1]>=30, (skeleton_data_numpy.shape)
            # skeleton data augmentation and view2 generation
            if self.apply_skeleton_augmentations:
                skeleton_data_numpy_1 = np.array(self.augment_skeleton(
                    torch.tensor(skeleton_data_numpy).float(), 
                    self.skeleton_augmentations))
                item["skeleton"]["view1"] = skeleton_data_numpy_1
                if self.skeleton_backbone != 'stgcn':
                    item["skeleton"]["view1"] = torch.FloatTensor(item["skeleton"]["view1"].squeeze().transpose(1, 2, 0))
                if self.n_views == 2:
                    # skeleton_data_numpy_2 = self.augment_skeleton_simple(
                    # skeleton_data_numpy, augemntation_apply=self.apply_augmentations
                    # )
                    skeleton_data_numpy_2 = np.array(self.augment_skeleton(
                        torch.tensor(skeleton_data_numpy).float(),
                        self.skeleton_augmentations))
                    
                    item["skeleton"]["view2"] = skeleton_data_numpy_2
                    if self.skeleton_backbone != 'stgcn':
                        item["skeleton"]["view2"] = torch.FloatTensor(
                        item["skeleton"]["view2"]
                        .squeeze()
                        .transpose(1, 2, 0)
                        )
        if "speech" in self.modalities:
            item["speech"] = {}
            start_frame = speech_start_frame * self.audio_sample_per_frame
            end_frame = start_frame + 2 * self.audio_sample_rate
            if self.debug_audio:
                audio_data = np.zeros(2 * self.audio_sample_rate)
            else: 
                if start_frame < 0:
                    padding = np.zeros(abs(start_frame))
                    start_frame = 0
                    audio_data = np.concatenate(
                        (padding, self.audio_dict[pair_speaker]['audio'][start_frame:end_frame])
                    )
                elif end_frame > len(self.audio_dict[pair_speaker]['audio']):
                    padding = np.zeros(end_frame - len(self.audio_dict[pair_speaker]['audio']))
                    end_frame = len(self.audio_dict[pair_speaker]['audio'])
                    audio_data = np.concatenate(
                        (self.audio_dict[pair_speaker]['audio'][start_frame:end_frame], padding)
                    )
                else:
                    audio_data = self.audio_dict[pair_speaker]['audio'][start_frame:end_frame]
                if len(audio_data) > number_speech_frames:
                    audio_data = audio_data[:number_speech_frames]
                elif len(audio_data) < number_speech_frames:
                    padding = np.zeros(number_speech_frames - len(audio_data))
                    audio_data = np.concatenate((audio_data, padding))

            item["speech"]["orig"] = audio_data  
            # speech data augmentation and view2 generation  
            if self.apply_speech_augmentations:
                speech_wave_form_1 = self.augment_audio(audio_data)
                item["speech"]["view1"] = speech_wave_form_1

                if self.n_views == 2:
                    speech_wave_form_2 = self.augment_audio(audio_data)                
                    item["speech"]["view2"] = speech_wave_form_2
        if self.skeleton_backbone != 'stgcn':
            item["skeleton"]["orig"] = torch.FloatTensor(
                item["skeleton"]["orig"]
                .squeeze()
                .transpose(1, 2, 0)
            )
        return item


if __name__ == '__main__':
    # test this class
    feeder = CABBFeeder(
        apply_skeleton_augmentations=False,
        task='segmentation',
        random_choose=False,
        window_size=72,
        phase='eval'
    )
    # get the first item
    item = feeder[0]
    print(item.keys())