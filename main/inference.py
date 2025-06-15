from wavlm.WavLM import WavLM, WavLMConfig
from hifigan.models import Generator as HiFiGAN
from hifigan.utils import AttrDict
from src.main.matcher import KNeighborsVC
from pathlib import Path
import json
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
import os
import torch, torchaudio
from sklearn.mixture import GaussianMixture
from LM import *
from LM_Infer import *
from LM_detokenizer import *
from LM_detokenizer_inf import *



def hifigan_wavlm(pretrained=True, progress=True, prematched=True, device='cuda') -> HiFiGAN:
    """ Load pretrained hifigan trained to vocode wavlm features. Optionally use weights trained on `prematched` data. """
    
    cp = Path.cwd().absolute()

    with open(cp / 'hifigan' / 'config_v1_wavlm.json') as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    device = torch.device(device)

    generator = HiFiGAN(h).to(device)

    if pretrained:
        if prematched:
     
            model_path = "/workspace/Kris/knn-vc/Denoising_Work/prematch_g_02500000.pt"
        else:

           
            model_path = "/workspace/Kris/knn-vc/Denoising_Work/prematch_g_02500000.pt"


        state_dict_g = torch.load(model_path, map_location=device)
        generator.load_state_dict(state_dict_g['generator'])

    generator.eval()
    generator.remove_weight_norm()
    print(f"[HiFiGAN] Generator loaded with {sum([p.numel() for p in generator.parameters()]):,d} parameters.")
    return generator, h




def load_expanded_set(matching_set_device):
    # if gender == 'male':
    #     expanded_set_path = "/workspace/Kris/Phoneme_Hallucinator/expanded_set-3.npy"
    # elif gender == 'female':
    #     expanded_set_path = "/workspace/Kris/Phoneme_Hallucinator/expanded_set_female.npy"
    # else:
    #     raise ValueError("Gender must be either 'male' or 'female'")
    # expanded_set_path ="/workspace/Kris/Phoneme_Hallucinator/feat_abx_30000/English_Normal_010_expanded.npy"
    # expanded_set_path ="/workspace/Kris/Phoneme_Hallucinator/expanded_sets/trained_PH_the_best_th_normal_3_30000.npy"  # the best 
    expanded_set_path =  "/workspace/Kris/knn-vc/Denoising_Work/expanded_set__perceiver_beta_500E_more_nor_Eng_thebest_30000.npy" #just PH
    
    return torch.tensor(np.load(expanded_set_path)).to(matching_set_device)



def wavlm_large(pretrained=True, progress=True, device='cuda') -> WavLM:
   
    if torch.cuda.is_available() == False:
        if str(device) != 'cpu':
            logging.warning(f"Overriding device {device} to cpu since no GPU is available.")
            device = 'cpu'

    
    wavlm_checkpoint_path = "WavLM-Large.pt"
    
   
    checkpoint = torch.load(wavlm_checkpoint_path, map_location=device)

    
    cfg = WavLMConfig(checkpoint['cfg'])
    device = torch.device(device)
    model = WavLM(cfg)
    if pretrained:
        model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    print(f"WavLM-Large loaded with {sum([p.numel() for p in model.parameters()]):,d} parameters.")
    return model





def knn_vc(pretrained=True, progress=True, prematched=True, device='cuda') -> KNeighborsVC:
    """ Load kNN-VC (WavLM encoder and HiFiGAN decoder). Optionally use vocoder trained on `prematched` data. """
    hifigan, hifigan_cfg = hifigan_wavlm(pretrained, progress, prematched, device)
    wavlm = wavlm_large(pretrained, progress, device)
    knnvc = KNeighborsVC(wavlm, hifigan, hifigan_cfg, device=device)
    return knnvc



knnvc_model = knn_vc(pretrained=True, progress=True, prematched=True, device='cuda')

# src feat extraction
src_wav_path = '/workspace/Kris/knn-vc/Denoising_Work/output/p232_005_ambulance.wav'
query_seq = knnvc_model.get_features(src_wav_path)

save_dir = '/workspace/Kris/knn-vc/App_feat/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


save_path = os.path.join(save_dir, file_name)

torch.save(query_seq, save_path)
print(f'query_seq saved successfully at {save_path}')
print(query_seq.shape)


#feat enhancing with LM
#Thai Weights
# model_path = "/workspace/Kris/knn-vc/speechlm_checkpoint_epoch_10.pt"
# tokenizer_path = "/workspace/Kris/knn-vc/tokenizer_epoch_10.pt"
#Eng Weights without Detokenizer
# model_path = '/workspace/Kris/knn-vc/English_LLM_ckpts/trained_speech_lm_final_en.pt'    #English
# tokenizer_path = '/workspace/Kris/knn-vc/English_LLM_ckpts/fitted_tokenizer_final_en.pt' #English


model_path = "/workspace/Kris/knn-vc/trained_speech_lm_with_decoder_final.pt" #with detokenizer, Eng
tokenizer_path = "/workspace/Kris/knn-vc/fitted_tokenizer_with_decoder_final.pt" #with detokenizer, Eng
input_dir = "/workspace/Kris/knn-vc/English_Speech_Feat/eng_alar_feat/converted_audio/"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
print(f"Initializing inference on device: {device}")
inferencer = SpeechLMInference(model_path, tokenizer_path, device=device, max_clusters=512)

# normal_features = torch.load("/workspace/Kris/knn-vc/czech/feat/czech_test/Czech_Normal_spk1_001.pt")
alar_features = torch.load("/workspace/Kris/knn-vc/English_Speech_Feat/eng_alar_feat/converted_audio/")

normal_features = normal_features.to(device)
alar_features = alar_features.to(device)

print(f"Loaded features - Normal: {normal_features.shape}, Alaryngeal: {alar_features.shape}")


inferencer.fit_tokenizer(torch.cat([normal_features, alar_features], dim=0))
# inferencer.fit_tokenizer(normal_features, min_clusters=181)

# # Analyze normal speech patterns
# print("Analyzing normal speech patterns...")
inferencer.analyze_normal_speech(normal_features)
tokens_normal =  inferencer.analyze_normal_speech(normal_features)

# Generate normal-like speech
print("Generating normal-like speech...")
generated_features = inferencer.generate_normal_like_speech(
    alar_features,
    seq_length=600,
    temperature=0.9,
    normal_bias=0.5
)

print(f"Input features shape: {alar_features.shape}")
print(f"Input dtype features shape: {alar_features.dtype}")
print(f"Generated features shape: {generated_features.shape}")
print('generated_features shape',generated_features.shape)




#ref feat extraction
ref_wav_paths = ['/workspace/Kris/knn-vc/English_Speech_&Feat/audios_abx_wav/normal/English_Normal_001.wav',] # this will be fixed one
matching_set = knnvc_model.get_matching_set(ref_wav_paths)

# load extended set according to matching set
expanded_set = load_expanded_set(matching_set.device)

#new matching set
matching_set = torch.cat([matching_set, expanded_set], 0)








 






# ref_wav_paths = ['/workspace/Kris/knn-vc/English_Speech_&Feat/audios_abx_wav/normal/English_Normal_002.wav',]

# knnvc_model = knn_vc(pretrained=True, progress=True, prematched=True, device='cuda')

# matching_set = knnvc_model.get_matching_set(ref_wav_paths)

# matching_set = matching_set.view(-1, matching_set.shape[-1])

# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)


# save_path = os.path.join(save_dir, file_name)

# torch.save(matching_set, save_path)
# print(f'Matching set saved successfully at {save_path}')
# print(matching_set.shape)



# expanded_set = load_expanded_set(matching_set.device)
# print(expanded_set.shape)
# matching_set = torch.cat([matching_set, expanded_set], 0)
# query_seq = knnvc_model.get_features(src_wav_path)
# print(query_seq.shape)
# print(matching_set.shape)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# knnvc_model = knnvc_model.to(device)
# query_seq = query_seq.to(device)
# matching_set = matching_set.to(device)
# src_feat_squeezed = src_feat_squeezed.to(device)

# src_feat_squeezed = src_feat_squeezed.float()
# matching_set = matching_set.float()
# print(matching_set.shape)
# print(src_feat_squeezed.shape)

# out_wav = knnvc_model.match(query_seq, matching_set)



# torchaudio.save('/workspace/Kris/knn-vc/Denoising_Work/output/Denoised_Eng_0002.wav', out_wav[None], 16000)
