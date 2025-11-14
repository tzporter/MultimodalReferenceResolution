import os
import numpy as np
import pandas as pd
import torch
from sklearn.svm import SVC
from torchvision import transforms, models
from PIL import Image
from transformers import AutoTokenizer, AutoModel
from model.skeleton_speech_models import GSSModel
from model.decouple_gcn_attn_sequential import Model as STGCN
from data.read_process_poses import process_poses
from feeders.gestures_feeder import load_audio_dict
from dialog_utils.prepare_gesture_referents_dataset import get_detailed_gestures
# from object_retrieval_pair import manual_implementation_object_retrieval
from gestures_forms_sim import measure_sim_gestures
from tqdm import tqdm
from gestures_forms_sim import process_skeleton, get_bone

import glob

pd.set_option("display.precision", 2)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def manual_implementation_object_retrieval(jointsformer,
                                           processed_keypoints_dict,
                                           mirrored_keypoints_dict,
                                           sekeleton_backbone,
                                           multimodal=True,
                                           bones=False,
                                           before_head_feats=False,
                                           gestures_and_speech_info=None,
                                           modalities=['skeleton', 'semantic'],
                                           audio_dict=None, model_type=None):
   """
   Manually implement the object retrieval functionality.
   """
   device = next(jointsformer.parameters()).device

   if gestures_and_speech_info is None:
      gestures_and_speech_info = get_detailed_gestures()

   # Reset indices and filter iconic gestures
   gestures_and_speech_info = gestures_and_speech_info.reset_index(drop=True)
#    gestures_and_speech_info['pair_speaker'] = gestures_and_speech_info['pair'] + '_' + gestures_and_speech_info['speaker']
#    gestures_and_speech_info = gestures_and_speech_info[gestures_and_speech_info['type'] == 'iconic'].reset_index(drop=True)

   jointsformer.eval()

   num_gestures = len(gestures_and_speech_info)
   if bones:
      feature_name = 'bone_features'
      feat_shape = 20
   elif not multimodal and not bones:
      feature_name = 'skeleton_features'
      feat_shape = 20 
   elif sekeleton_backbone == 'stgcn' and multimodal:
      feature_name = 'stgcn_features'
      feat_shape = 20  
   else:
      if model_type is not None:
         feature_name = model_type
      else:
         feature_name = 'transformer_features'
      feat_shape = 20  

   gestures_and_speech_info[feature_name] = [np.zeros(feat_shape) for _ in range(num_gestures)]
   gestures_and_speech_info['keypoints_features'] = [np.zeros(81) for _ in range(num_gestures)]
   gestures_and_speech_info['semantic_features'] = [np.zeros(16) for _ in range(num_gestures)]
   gestures_and_speech_info['speech_features'] = gestures_and_speech_info.apply(lambda x: np.zeros(16), axis=1)

   # Process each gesture
   error_count = 0
   for row in tqdm(gestures_and_speech_info.itertuples(), total=num_gestures):
      i = row.Index
      start_frame = row.start_frame
      end_frame = row.end_frame
      ch = row.chapter
      ch_start = f"{ch}_{start_frame}"
      file = ch_start
    #   
    #   if end_frame - start_frame < 2:
    #      start_frame -= 2
    #      end_frame += 2
    
      if end_frame - start_frame > 72:
         # Choose the middle 72 frames; this is because the model expects 72 frames
         shift = (end_frame - start_frame - 72) // 2
         start_index = shift
         end_index = start_index + 72
      else:
         start_index = 0
         end_index = end_frame - start_frame
      try:
        skel = processed_keypoints_dict[file][:, start_index:end_index, :, :]
        mirrored_skel = mirrored_keypoints_dict[file][:, start_index:end_index, :, :]
      except Exception as e:
         error_count += 1
         # print(e)
         # print('Error in loading the keypoints for the gesture:', file)
         continue
    #   if speaker == 'B':
        #  skel, mirrored_skel = mirrored_skel, skel

      skel = process_skeleton(skel, mirrored_skel)
      bone_skel = get_bone(skel)
      skel_tensor = torch.tensor(skel, device=device, dtype=torch.float32).squeeze()
      bone_tensor = torch.tensor(bone_skel, device=device, dtype=torch.float32)

      if sekeleton_backbone != 'stgcn':
         skel_tensor = skel_tensor.permute(0, 2, 3, 1)
      else:
         skel_tensor = skel_tensor.unsqueeze(-1)
      if bones:
         skel_tensor = bone_tensor

    #   utterance = row.processed_utterance
      with torch.no_grad():
         mm_outputs = jointsformer(
               skel_tensor, eval=True, utterances=[None],
               features=modalities, before_head_feats=before_head_feats,
            #    speech_waveform=speech_waveform
         )
         if mm_outputs.get('one_branch_cross_modal', False):
               skeleton_feats = mm_outputs['skeleton_features'][0]
         else:
               skeleton_feats = torch.cat([mm_outputs['skeleton_features'][0], mm_outputs['skeleton_features'][1]], dim=0)
         skeleton_feats = torch.nn.functional.normalize(skeleton_feats, dim=0)

         if mm_outputs.get('semantic_features') is not None:
               semantic_feats = mm_outputs['semantic_features'].squeeze()
               semantic_feats = torch.nn.functional.normalize(semantic_feats, dim=0)
               gestures_and_speech_info.at[i, 'semantic_features'] = semantic_feats.cpu().numpy()

         if mm_outputs.get('speech_features') is not None:
               speech_feats = mm_outputs['speech_features'].squeeze()
               speech_feats = torch.nn.functional.normalize(speech_feats, dim=0)
               gestures_and_speech_info.at[i, 'speech_features'] = speech_feats.cpu().numpy()
      print(skeleton_feats.shape)
      skeleton_feats = skeleton_feats.reshape(1, -1)
      gestures_and_speech_info.at[i, feature_name] = skeleton_feats[0].cpu().numpy()

      if sekeleton_backbone == 'stgcn':
         skel_tensor = skel_tensor.squeeze(-1).permute(0, 2, 3, 1)
      skel_mean = skel_tensor.mean(dim=1).view(skel_tensor.size(0), -1)
      skel_norm = torch.nn.functional.normalize(skel_mean, dim=1)
      gestures_and_speech_info.at[i, 'keypoints_features'] = skel_norm[0].cpu().numpy()
   return None, gestures_and_speech_info


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask
def load_keypoints_dict():
   # read all poses in f'data/final_poses/poses_{pair}_synced_pp{speaker}.npy'
   error_ticker = 0
   processed_keypoints_dict = {}
   mirrored_keypoints_dict = {}
   for file in glob.glob('selected_poses/*.npy'):
      # file format = data/selected_poses/poses_pair04_synced_ppA.npy 
    #   pair = file.split('/')[-1].split('_')[1]
    #   speaker = file.split('/')[-1].split('_')[-1].split('.')[0][-1]
    #   pair_speaker = f"{pair}_{speaker}"
      name = '-'.join(file.split('/')[-1].split('_')[1:-1])  # get filename without path and extension
      name = name.split('-')
      chapter = name[1]
      start_frame = name[3]
      ch_start = f"{chapter}_{start_frame}"
      try:
         processed_keypoints_dict[ch_start], mirrored_keypoints_dict[ch_start] = process_poses(np.load(file)) #np.load(keypoints_path)
      except Exception as e:
          error_ticker += 1
         # print(e)
         # print('Error in loading the keypoints for the pair:', file)
         # continue
   print(f"Total keypoint loading errors: {error_ticker}")
   return processed_keypoints_dict, mirrored_keypoints_dict
# def get_embeddings(sentences, model, tokenizer):
#     encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(DEVICE)
#     with torch.no_grad():
#         model_output = model(**encoded_input)
#     return mean_pooling(model_output, encoded_input['attention_mask'])

# def load_models():

#     # BERT model and tokenizer
#     tokenizer = AutoTokenizer.from_pretrained("GroNLP/bert-base-dutch-cased")
#     bert_model = AutoModel.from_pretrained("GroNLP/bert-base-dutch-cased", output_hidden_states=True).to(DEVICE)
#     bert_model.eval()

#     return bert_model, tokenizer

def load_pretrained_model(model_params=None):
    model_path = model_params['model_weights'] 
    # remove the model weights from the model params
    del model_params['model_weights']
    pretrained_model = GSSModel(**model_params)
    weights_dict = torch.load(model_path, map_location=torch.device('cpu'))
    for key in pretrained_model.state_dict().keys():
        pretrained_model.state_dict()[key].copy_(weights_dict['state_dict']['model.'+key])
    return pretrained_model.to(DEVICE)


def main():
    processed_keypoints_dict, mirrored_keypoints_dict = load_keypoints_dict()
    speakers_pairs = processed_keypoints_dict.keys()
    # remove pairs without 'pair' --> this is to focus on CABB-S pairs
    speakers_pairs = [speaker_pair for speaker_pair in speakers_pairs if 'pair' in speaker_pair]

    # audio_dict = {} #load_audio_dict(speakers_pairs, audio_path='data/audio_files/{}_synced_pp{}.wav', audio_sample_rate=16000)
    gestures_info_exploded = pd.read_csv('./full_segments.csv')
    # gestures_info['round'] = gestures_info['round'].astype(int)
    # gestures_info['object'] = gestures_info['object'].astype(int)
    # bert_model, tokenizer = load_models()
    # gestures_info = gestures_info[gestures_info['type'] == 'iconic'].reset_index(drop=True)

    # get semantic embeddings
    # gestures_info['semantic_embeddings'] = gestures_info['processed_utterance'].apply(
    #     lambda x: get_embeddings([x], bert_model, tokenizer).squeeze(0).cpu().numpy()
    # )
    
    # gestures_info['referent'].unique()
    # remove -REP, -rep, -WF, -wf from referent
    # to_remove = ['-REP', '-rep', '-WF', '-wf', 'ref', 'REF']
    # ***replace referrent with is_present***
    # gestures_info['referent_clean'] = gestures_info['referent'].apply(lambda x: x if not x.endswith(tuple(to_remove)) else x.split('-')[0])
    # explode the referent_clean column with '+' as separator
    # gestures_info_exploded = gestures_info.assign(referent_clean=gestures_info['referent_clean'].str.split('+')).explode('referent_clean')
    # gestures_info_exploded['referent_clean'] = gestures_info_exploded['referent_clean'].str.strip()
    # replace - with 0 in referent_clean
    # gestures_info_exploded['referent_clean'] = gestures_info_exploded['referent_clean'].str.replace('-', '0')
    # gestures_info_exploded['referent_clean'] = gestures_info_exploded['referent_clean'].str.replace('\'', '')
    # remove undecided from referent_clean
    # gestures_info_exploded = gestures_info_exploded[gestures_info_exploded['referent_clean'] != 'undecided']
    # gestures_info_exploded['referent_clean'] = gestures_info_exploded.apply(lambda x: f'{17}' if 'general' in x['referent_clean'] else x['referent_clean'], axis=1)
    # gestures_info_exploded['referent_clean'] = gestures_info_exploded.apply(lambda x: f'{0}' if 'main' in x['referent_clean'] else x['referent_clean'], axis=1)
        
    #TODO: load the models from our work properly. For now, we are using hardcoded paths
    model_types = {
        # 'multimodal-x-skeleton-semantic': 
        #     {'modalities': ['skeleton', 'semantic'], 'hidden_dim': 256, 'cross_modal':True,'attentive_pooling':False, 'model_weights':  'pretrained_models/multimodal-x_skeleton_text_correlation=0.30.ckpt'},
        # 'multimodal-skeleton-semantic': 
        #      {'modalities': ['skeleton', 'semantic'], 'hidden_dim': 256, 'cross_modal':False,'attentive_pooling':False, 'model_weights':  'pretrained_models/multimodal_skeleton_text_correlation=0.29.ckpt'},
        'unimodal_skeleton': 
            {'modalities': ['skeleton'], 'hidden_dim': 256, 'cross_modal':False, 'attentive_pooling':False, 'model_weights':  'pretrained_models/unimodal_correlation=0.22.ckpt'},
    }
    # all_model_results = [] 
    
    for model_type in model_types.keys():
        print(f'Loaded model {model_type} with modalities {model_types[model_type]["modalities"]} and weights from {model_types[model_type]["model_weights"]}')
        pretrained_model = load_pretrained_model(model_params=model_types[model_type])
        pretrained_model.eval()
        pretrained_model.to(DEVICE)
        # spearman_correlation, difference, combined_gestures_form = measure_sim_gestures(pretrained_model , multimodal=True,processed_keypoints_dict=processed_keypoints_dict, mirrored_keypoints_dict=mirrored_keypoints_dict, modalities=model_types[model_type]['modalities'],audio_dict=audio_dict)
        _, gestures_info_exploded = manual_implementation_object_retrieval(pretrained_model , processed_keypoints_dict, mirrored_keypoints_dict, sekeleton_backbone='jointsformer', gestures_and_speech_info=gestures_info_exploded, modalities=model_types[model_type]['modalities'], audio_dict={},model_type=model_type)
        # rename the column 'transformer_features' to the model type
        gestures_info_exploded.rename(columns={'transformer_features': model_type}, inplace=True)

        # all_model_results.append(retrieval_results)
        

    # combined_retrieval_results = pd.concat(all_model_results, ignore_index=True)
    # save the combined retrieval results
    # combined_retrieval_results.to_csv('results/combined_retrieval_results_whole_object_level.csv')
          
   
    # random skeleton features (baseline)
    # gestures_info_exploded['random_skeleton_features'] = gestures_info_exploded['multimodal-skeleton-semantic'].apply(
    #     lambda x: np.random.rand(x.shape[0], 1)
    # )
    
    # gestures_info_exploded['semantic+multimodal-x'] = gestures_info_exploded.apply( lambda x: np.concatenate([x['multimodal-x-skeleton-semantic'], x['semantic_embeddings']]), axis=1)
    # gestures_info_exploded['semantic+multimodal'] = gestures_info_exploded.apply( lambda x: np.concatenate([x['multimodal-skeleton-semantic'], x['semantic_embeddings']]), axis=1)
    # gestures_info_exploded['semantic+unimodal'] = gestures_info_exploded.apply( lambda x: np.concatenate([x['unimodal_skeleton'], x['semantic_embeddings']]), axis=1)
    

    gestures_info_exploded.to_pickle('data/parent_gesture_embeddings.pkl')
    

if __name__ == "__main__":
    main()
