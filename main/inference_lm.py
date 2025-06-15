from models.wavlm.WavLM import WavLM, WavLMConfig
from models.hifigan.models import Generator as HiFiGAN
from models.hifigan.utils import AttrDict
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
import argparse


def parse_arguments():
    """Parse command line arguments for gender and language selection"""
    parser = argparse.ArgumentParser(description='Voice conversion with gender and language options')
    parser.add_argument('--gender', type=str, choices=['male', 'female'], default='male',
                        help='Gender for voice conversion (male or female)')
    parser.add_argument('--language', type=str, choices=['english', 'thai'], default='english',
                        help='Language for voice conversion (english or thai)')
    parser.add_argument('--input', type=str, default='/mnt/d/Senior_Project_KMUTT/ASPEAK/0001_0001.wav',
                        help='Path to the input audio file')
    parser.add_argument('--ref', type=str, default='/mnt/d/Senior_Project_KMUTT/ASPEAK/normal_1.wav',
                        help='Path to the reference audio file')
    parser.add_argument('--output', type=str, default='/mnt/d/Senior_Project_KMUTT/ASPEAK/final_output.wav',
                        help='Path to save the output audio file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    return parser.parse_args()


def hifigan_wavlm(pretrained=True, progress=True, prematched=True, device='cuda') -> HiFiGAN:
    """ Load pretrained hifigan trained to vocode wavlm features. Optionally use weights trained on `prematched` data. """
    
    cp = Path.cwd().absolute()

    with open(cp / 'models' / 'hifigan' / 'config_v1_wavlm.json') as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    device = torch.device(device)

    generator = HiFiGAN(h).to(device)

    if pretrained:
        if prematched:
            model_path = cp / "models" / "hifigan" / "prematch_g_02500000.pt"
        else:
            model_path = cp / "models" / "hifigan" / "prematch_g_02500000.pt"

        state_dict_g = torch.load(model_path, map_location=device)
        generator.load_state_dict(state_dict_g['generator'])

    generator.eval()
    generator.remove_weight_norm()
    #print(f"[HiFiGAN] Generator loaded with {sum([p.numel() for p in generator.parameters()]):,d} parameters.")
    return generator, h


def load_expanded_set(gender, language, matching_set_device):
    """Load expanded set based on gender and language"""
    base_path = Path.cwd().absolute() / "models" / "expanded_sets"
    
    # Select the appropriate dataset based on gender and language
    if gender == 'male' and language == "english":
        expanded_set_path = base_path / "/home/nuos/ASPEAK/src/expanded_set/alar_eng_0_new_spkr.npy"
    elif gender == 'female' and language == "english":
        expanded_set_path = base_path / "/home/nuos/ASPEAK/src/expanded_set/matching_set_016_002_michelle_Expanded.npy"
    elif gender == 'male' and language == "thai":
        expanded_set_path = base_path / "/home/nuos/ASPEAK/src/expanded_set/expanded_set_male.npy"
    elif gender == 'female' and language == "thai":
        expanded_set_path = base_path / "/home/nuos/ASPEAK/src/expanded_set/expanded_set_female.npy"
    else:
        # Fallback to default dataset
        # expanded_set_path = base_path / "expanded_set__perceiver_beta_500E_more_nor_Eng_thebest_30000.npy"
        print(f"Warning: Using default expanded set for {gender}/{language}")
    
    # Check if the file exists, otherwise fall back to default
    if not expanded_set_path.exists():
        print(f"Warning: Expanded set file not found at {expanded_set_path}")
        print(f"Falling back to default expanded set")
        expanded_set_path = base_path / "expanded_set__perceiver_beta_500E_more_nor_Eng_thebest_30000.npy"
    
    #print(f"Loading expanded set from: {expanded_set_path}")
    return torch.tensor(np.load(expanded_set_path)).to(matching_set_device)


def wavlm_large(pretrained=True, progress=True, device='cuda') -> WavLM:
   
    if torch.cuda.is_available() == False:
        if str(device) != 'cpu':
            logging.warning(f"Overriding device {device} to cpu since no GPU is available.")
            device = 'cpu'

    wavlm_checkpoint_path = Path.cwd().absolute() / "models" / "wavlm" / "WavLM-Large.pt"
    
    checkpoint = torch.load(wavlm_checkpoint_path, map_location=device)

    cfg = WavLMConfig(checkpoint['cfg'])
    device = torch.device(device)
    model = WavLM(cfg)
    if pretrained:
        model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    # print(f"WavLM-Large loaded with {sum([p.numel() for p in model.parameters()]):,d} parameters.")
    return model


def knn_vc(pretrained=True, progress=True, prematched=True, device='cuda') -> KNeighborsVC:
    """ Load kNN-VC (WavLM encoder and HiFiGAN decoder). Optionally use vocoder trained on `prematched` data. """
    hifigan, hifigan_cfg = hifigan_wavlm(pretrained, progress, prematched, device)
    wavlm = wavlm_large(pretrained, progress, device)
    knnvc = KNeighborsVC(wavlm, hifigan, hifigan_cfg, device=device)
    return knnvc


def main():
    # Parse arguments
    args = parse_arguments()
    gender = args.gender
    language = args.language
    input_path = args.input
    ref_path = args.ref
    output_path = args.output
    device_name = args.device
    
    # Display configuration
    # print(f"Configuration:")
    # print(f"- Gender: {gender}")
    # print(f"- Language: {language}")
    # print(f"- Input audio: {input_path}")
    # print(f"- Reference audio: {ref_path}")
    # print(f"- Output path: {output_path}")
    # print(f"- Device: {device_name}")
    
    # Set device
    device = torch.device(device_name if torch.cuda.is_available() and device_name == 'cuda' else 'cpu')
    # print(f"Using device: {device}")
    
    # Initialize KNN-VC model
    knnvc_model = knn_vc(pretrained=True, progress=True, prematched=True, device=device)
    
    # Extract source features
    # print(f"Extracting features from source audio: {input_path}")
    query_seq = knnvc_model.get_features(input_path)
    
    # Save extracted features
    save_dir = '/mnt/d/Senior_Project_KMUTT/ASPEAK/output_LM_Detokenizer/'
    file_name = "query_seq.pt"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_path = os.path.join(save_dir, file_name)
    torch.save(query_seq, save_path)
    # print(f'Query sequence saved successfully at {save_path}')
    # print(f'Query sequence shape: {query_seq.shape}')
    
    # Get reference audio features
    ref_wav_paths = [ref_path]
    matching_set = knnvc_model.get_matching_set(ref_wav_paths)
    matching_set = matching_set.view(-1, matching_set.shape[-1])
    
    # Load expanded set based on gender and language
    expanded_set = load_expanded_set(gender, language, matching_set.device)
    print(f'Expanded set shape: {expanded_set.shape}')
    
    # Combine reference and expanded set
    matching_set = torch.cat([matching_set, expanded_set], 0)
    print(f'Combined matching set shape: {matching_set.shape}')
    
    # Ensure tensors are on the correct device and type
    knnvc_model = knnvc_model.to(device)
    matching_set = matching_set.to(device)
    enhanced_src_feat = query_seq.to(device)
    
    enhanced_src_feat = query_seq.float()
    matching_set = matching_set.float()
    
    # Generate output with fuzzy spectral attention
    print(f"Generating output audio with {gender} {language} voice...")
    out_wav = knnvc_model.match_with_fuzzy_spectral_attention(enhanced_src_feat, matching_set)
    
    # Save output audio
    print(f"Saving output to {output_path}")
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    torchaudio.save(output_path, out_wav[None], 16000)
    print(f"Conversion complete! Output saved to {output_path}")


if __name__ == "__main__":
    main()

# from wavlm.WavLM import WavLM, WavLMConfig
# from hifigan.models import Generator as HiFiGAN
# from hifigan.utils import AttrDict
# from matcher import KNeighborsVC
# from pathlib import Path
# import json
# import torch
# from torch import Tensor
# import torch.nn as nn
# import torch.nn.functional as F
# import logging
# import numpy as np
# import os
# import torch, torchaudio
# from sklearn.mixture import GaussianMixture




# def hifigan_wavlm(pretrained=True, progress=True, prematched=True, device='cuda') -> HiFiGAN:
#     """ Load pretrained hifigan trained to vocode wavlm features. Optionally use weights trained on `prematched` data. """
    
#     cp = Path.cwd().absolute()

#     with open(cp / 'hifigan' / 'config_v1_wavlm.json') as f:
#         data = f.read()
#     json_config = json.loads(data)
#     h = AttrDict(json_config)
#     device = torch.device(device)

#     generator = HiFiGAN(h).to(device)

#     if pretrained:
#         if prematched:
#             model_path = "/mnt/d/Senior_Project_KMUTT/ASPEAK/prematch_g_02500000.pt"
#         else:
#             model_path = "/mnt/d/Senior_Project_KMUTT/ASPEAK/prematch_g_02500000.pt"

#         state_dict_g = torch.load(model_path, map_location=device)
#         generator.load_state_dict(state_dict_g['generator'])

#     generator.eval()
#     generator.remove_weight_norm()
#     print(f"[HiFiGAN] Generator loaded with {sum([p.numel() for p in generator.parameters()]):,d} parameters.")
#     return generator, h


# ######final app#########

# def load_expanded_set(matching_set_device):
#     if gender == 'male' & language == "english":
#         expanded_set_path = "/mnt/d/Senior_Project_KMUTT/ASPEAK/expanded_set__perceiver_beta_500E_more_nor_Eng_thebest_30000.npy"
#     elif gender == 'female' & language == "english":
#         expanded_set_path = "/mnt/d/Senior_Project_KMUTT/ASPEAK/expanded_set__perceiver_beta_500E_more_nor_Eng_thebest_30000.npy"
#     elif gender == 'male' & language == "thai":
#         expanded_set_path = "/mnt/d/Senior_Project_KMUTT/ASPEAK/expanded_set__perceiver_beta_500E_more_nor_Eng_thebest_30000.npy"
#     elif gender == 'female' & language == "thai":
#         expanded_set_path = "/mnt/d/Senior_Project_KMUTT/ASPEAK/expanded_set__perceiver_beta_500E_more_nor_Eng_thebest_30000.npy"
#     else:
#         raise ValueError("Gender must be either 'male' or 'female'")
    
   
#     return torch.tensor(np.load(expanded_set_path)).to(matching_set_device)

# # def load_expanded_set(matching_set_device):
    
   
# #     expanded_set_path = "/mnt/d/Senior_Project_KMUTT/ASPEAK/expanded_set__perceiver_beta_500E_more_nor_Eng_thebest_30000.npy"
   
# #     return torch.tensor(np.load(expanded_set_path)).to(matching_set_device)



# def wavlm_large(pretrained=True, progress=True, device='cuda') -> WavLM:
   
#     if torch.cuda.is_available() == False:
#         if str(device) != 'cpu':
#             logging.warning(f"Overriding device {device} to cpu since no GPU is available.")
#             device = 'cpu'

    
#     wavlm_checkpoint_path = "WavLM-Large.pt"
    
   
#     checkpoint = torch.load(wavlm_checkpoint_path, map_location=device)

    
#     cfg = WavLMConfig(checkpoint['cfg'])
#     device = torch.device(device)
#     model = WavLM(cfg)
#     if pretrained:
#         model.load_state_dict(checkpoint['model'])
#     model = model.to(device)
#     model.eval()
#     print(f"WavLM-Large loaded with {sum([p.numel() for p in model.parameters()]):,d} parameters.")
#     return model





# def knn_vc(pretrained=True, progress=True, prematched=True, device='cuda') -> KNeighborsVC:
#     """ Load kNN-VC (WavLM encoder and HiFiGAN decoder). Optionally use vocoder trained on `prematched` data. """
#     hifigan, hifigan_cfg = hifigan_wavlm(pretrained, progress, prematched, device)
#     wavlm = wavlm_large(pretrained, progress, device)
#     knnvc = KNeighborsVC(wavlm, hifigan, hifigan_cfg, device=device)
#     return knnvc



# knnvc_model = knn_vc(pretrained=True, progress=True, prematched=True, device='cuda')

# # src feat extraction
# src_wav_path = '/mnt/d/Senior_Project_KMUTT/ASPEAK/0001_0001.wav'
# query_seq = knnvc_model.get_features(src_wav_path)

# save_dir = '/mnt/d/Senior_Project_KMUTT/ASPEAK/output_LM_Detokenizer/'
# file_name = "query_seq.pt"
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)


# save_path = os.path.join(save_dir, file_name)

# torch.save(query_seq, save_path)
# print(f'query_seq saved successfully at {save_path}')
# print(query_seq.shape)


# #feat enhancing with LM
# #Thai Weights
# # model_path = "/workspace/Kris/knn-vc/speechlm_checkpoint_epoch_10.pt"
# # tokenizer_path = "/workspace/Kris/knn-vc/tokenizer_epoch_10.pt"
# #Eng Weights without Detokenizer
# # model_path = '/workspace/Kris/knn-vc/English_LLM_ckpts/trained_speech_lm_final_en.pt'    #English
# # tokenizer_path = '/workspace/Kris/knn-vc/English_LLM_ckpts/fitted_tokenizer_final_en.pt' #English



# # def load_speechlm_model(model_path, tokenizer_path, device='cuda'):
# #     """Load trained SpeechLM model and tokenizer"""
# #     # Load tokenizer
# #     tokenizer = SELMEncoder.load_tokenizer(tokenizer_path)
    
# #     # Load model checkpoint
# #     checkpoint = torch.load(model_path, map_location=device)
# #     feature_dim = tokenizer.feature_dim
    
# #     # Initialize model
# #     model = SpeechLMWithDecoder(
# #         vocab_size=tokenizer.n_clusters,
# #         feature_dim=feature_dim
# #     ).to(device)
    
# #     # Load state dict
# #     model.load_state_dict(checkpoint['model_state_dict'])
# #     model.eval()
    
# #     return model, tokenizer


# # def inference_speechlm(model, tokenizer, input_features, sequence_length=500, device='cuda'):
# #     model.eval()
# #     with torch.no_grad():
# #         print(f"Input features shape: {input_features.shape}")
# #         input_tokens = tokenizer._encode_single(input_features)
# #         print(f"Encoded tokens shape: {input_tokens.shape}")
        
# #         reconstructed_features = []
# #         for i in range(0, len(input_tokens), sequence_length):
# #             sequence = input_tokens[i:i + sequence_length]
# #             if len(sequence) < sequence_length:
# #                 padding = sequence_length - len(sequence)
# #                 sequence = torch.cat([sequence, torch.zeros(padding, dtype=torch.long)])
            
# #             sequence = sequence.unsqueeze(0).to(device)
# #             print(f"Processing sequence batch shape: {sequence.shape}")
            
# #             try:
# #                 _, decoded = model(sequence, return_decoded=True)
# #                 print(f"Decoded features shape: {decoded.shape}")
# #                 reconstructed_features.append(decoded[0, :len(input_tokens[i:i + sequence_length])])
# #             except Exception as e:
# #                 print(f"Error during model inference: {str(e)}")
# #                 raise
        
# #         try:
# #             reconstructed_features = torch.cat(reconstructed_features, dim=0)
# #             print(f"Final reconstructed features shape: {reconstructed_features.shape}")
# #             return reconstructed_features
# #         except Exception as e:
# #             print(f"Error during feature concatenation: {str(e)}")
# #             print(f"Number of feature chunks: {len(reconstructed_features)}")
# #             raise

# # def process_feature_file(model, tokenizer, feature_path, output_path=None, device='cuda'):
# #     print(f"Processing file: {feature_path}")
# #     features = torch.load(feature_path, map_location=device)
# #     print(f"Loaded features shape: {features.shape}")
    
# #     features = features.float()
# #     reconstructed = inference_speechlm(model, tokenizer, features, device=device)
    
# #     if output_path:
# #         torch.save(reconstructed, output_path)
# #         print(f"Saved reconstructed features to: {output_path}")
    
# #     return reconstructed

# # def batch_process_directory(model_path, tokenizer_path, input_dir, output_dir, device='cuda'):
# #     """Process all feature files in a directory"""
# #     # Load model and tokenizer
# #     model, tokenizer = load_speechlm_model(model_path, tokenizer_path, device)
    
# #     # Create output directory
# #     output_dir = Path(output_dir)
# #     output_dir.mkdir(parents=True, exist_ok=True)
    
# #     # Process all .pt files
# #     input_files = list(Path(input_dir).glob("*.pt"))
# #     for input_file in input_files:
# #         output_file = output_dir / input_file.name
# #         process_feature_file(model, tokenizer, input_file, output_file, device)

        

# # model_path = "speechlm_decoder_checkpoint_epoch_30.pt" #with detokenizer, Eng
# # tokenizer_path = "tokenizer_with_decoder_epoch_30.pt" #with detokenizer, Eng
# # input_dir = "/mnt/d/Senior_Project_KMUTT/ASPEAK/output_LM_Detokenizer/"

# # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
# # print(f"Initializing inference on device: {device}")
# # output_dir = "/mnt/d/Senior_Project_KMUTT/ASPEAK/enhanced_feat_LM"


# #enhanced src feat from LM
# #enhanced_src_feat = batch_process_directory(model_path, tokenizer_path, input_dir, output_dir)



# ref_wav_paths = ['/mnt/d/Senior_Project_KMUTT/ASPEAK/normal_1.wav',]

# knnvc_model = knn_vc(pretrained=True, progress=True, prematched=True, device='cuda')

# matching_set = knnvc_model.get_matching_set(ref_wav_paths)

# matching_set = matching_set.view(-1, matching_set.shape[-1])


# expanded_set = load_expanded_set(matching_set.device)
# print(expanded_set.shape)
# matching_set = torch.cat([matching_set, expanded_set], 0)

# print(query_seq.shape)
# print(matching_set.shape)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# knnvc_model = knnvc_model.to(device)
# matching_set = matching_set.to(device)
# enhanced_src_feat = query_seq.to(device)

# enhanced_src_feat = query_seq.float()
# matching_set = matching_set.float()
# print(matching_set.shape)
# print(enhanced_src_feat.shape)

# out_wav = knnvc_model.match_with_fuzzy_spectral_attentio(enhanced_src_feat, matching_set)



# torchaudio.save('/mnt/d/Senior_Project_KMUTT/ASPEAK/final_output_thai_0001_0001.wav', out_wav[None], 16000)
