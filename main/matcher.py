import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from models.hifigan.models import Generator as HiFiGAN
from models.hifigan.utils import AttrDict
from torch import Tensor
from torchaudio.sox_effects import apply_effects_tensor
from models.wavlm.WavLM import WavLM
from src.main.knnvc_utils import generate_matrix_from_index
from sklearn.cluster import DBSCAN
from torch.nn.functional import pad
from sklearn.cluster import SpectralClustering
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import distance
from scipy.signal import savgol_filter
from scipy import interpolate

from scipy.spatial import distance

from scipy.linalg import svd


SPEAKER_INFORMATION_LAYER = 6
SPEAKER_INFORMATION_WEIGHTS = generate_matrix_from_index(SPEAKER_INFORMATION_LAYER)





path = "/workspace/Kris/Phoneme_Hallucinator/mapped_feats/matched_features.pt"


def fuzzy_c_means(data, c, m, error, maxiter):
    n = data.shape[0]
    u = np.random.rand(n, c)
    row_sums = u.sum(axis=1)
    u = u / row_sums[:, np.newaxis]

    jm = []

    for iteration in range(maxiter):
        um = u ** m
        centers = np.dot(um.T, data) / np.sum(um.T, axis=1, keepdims=True)

        d = distance.cdist(data, centers, metric='euclidean')
        d = np.fmax(d, np.finfo(np.float64).eps)

        jm.append(np.sum((d ** 2) * um))

        if iteration > 0 and abs(jm[iteration] - jm[iteration - 1]) < error:
            break

        d_inv = 1.0 / d
        power = 2.0 / (m - 1)
        d_inv_power = d_inv ** power
        u = d_inv_power / np.sum(d_inv_power, axis=1, keepdims=True)

    return centers, u, jm, d

def compute_temporal_weights(data, window_size=5):
         
    n_frames = data.shape[0]
    temporal_weights = np.zeros((n_frames, n_frames))

    for i in range(n_frames):
        # Define temporal neighborhood
        start = max(0, i - window_size)
        end = min(n_frames, i + window_size + 1)

        # Create weights based on temporal distance
        for j in range(start, end):
            if i != j:
                # Gaussian weighting based on temporal distance
                temporal_weights[i, j] = np.exp(-0.5 * ((i - j) / (window_size/2))**2)

    # Normalize weights
    row_sums = temporal_weights.sum(axis=1)
    temporal_weights = temporal_weights / (row_sums[:, np.newaxis] + 1e-10)

    return temporal_weights

def spatial_fuzzy_c_means(data, c, m, error, maxiter, lambda_param=0.3):
    """Fuzzy C-Means with Spatial (Temporal) Constraints"""
    n = data.shape[0]
    u = np.random.rand(n, c)
    row_sums = u.sum(axis=1)
    u = u / row_sums[:, np.newaxis]

    # Compute temporal neighborhood weights
    temporal_weights = compute_temporal_weights(data)

    jm = []

    for iteration in range(maxiter):
        # Calculate cluster centers using standard FCM
        um = u ** m
        centers = np.dot(um.T, data) / np.sum(um.T, axis=1, keepdims=True)

        # Compute distances to cluster centers
        d = distance.cdist(data, centers, metric='euclidean')
        d = np.fmax(d, np.finfo(np.float64).eps)

        # Calculate objective function
        jm.append(np.sum((d ** 2) * um))

        if iteration > 0 and abs(jm[iteration] - jm[iteration - 1]) < error:
            break

        # Standard FCM membership update
        d_inv = 1.0 / d
        power = 2.0 / (m - 1)
        d_inv_power = d_inv ** power
        u_fcm = d_inv_power / np.sum(d_inv_power, axis=1, keepdims=True)

        # Calculate spatial (temporal) term
        h = np.zeros_like(u_fcm)
        for i in range(n):
            for j in range(c):
                # Weight memberships by temporal neighborhood
                h[i, j] = np.sum(temporal_weights[i] * u_fcm[:, j])

        # Combine standard FCM memberships with temporal information
        u = (1 - lambda_param) * u_fcm + lambda_param * h

        # Normalize memberships
        row_sums = u.sum(axis=1)
        u = u / row_sums[:, np.newaxis]

    return centers, u, jm, d



def coral_transform(source, target):
    """
    Apply CORAL transformation to align source domain with target domain.
    
    Args:
        source (torch.Tensor): Source domain features (n_samples, n_features)
        target (torch.Tensor): Target domain features (m_samples, n_features)
        
    Returns:
        torch.Tensor: Transformed source features aligned with target domain
    """
    # Convert to float32 if needed
    source = source.float()
    target = target.float()
    
    # Calculate covariance matrices
    n_s = source.shape[0]
    n_t = target.shape[0]
    
    # Center the data
    source_centered = source - torch.mean(source, dim=0, keepdim=True)
    target_centered = target - torch.mean(target, dim=0, keepdim=True)
    
    # Compute covariance matrices
    source_cov = (source_centered.T @ source_centered) / (n_s - 1)
    target_cov = (target_centered.T @ target_centered) / (n_t - 1)
    
    # Add small regularization for numerical stability
    eps = 1e-5
    source_cov += torch.eye(source_cov.shape[0], device=source_cov.device) * eps
    target_cov += torch.eye(target_cov.shape[0], device=target_cov.device) * eps
    
    # Compute whitening and re-coloring transformation
    s_sqrt_inv = torch.linalg.inv(torch.linalg.cholesky(source_cov))
    t_sqrt = torch.linalg.cholesky(target_cov)
    
    # Apply transformation: first whiten source data, then re-color with target covariance
    transform_matrix = t_sqrt @ s_sqrt_inv
    
    # Apply transformation to centered source data
    transformed_source = source_centered @ transform_matrix
    
    # Add target mean to match first-order statistics as well
    target_mean = torch.mean(target, dim=0, keepdim=True)
    transformed_source = transformed_source + target_mean
    
    return transformed_source


def multi_head_attention(query, key, value, num_heads):
    """ Perform multi-head scaled dot product attention """
    d_k = query.size(-1) // num_heads  # Dimension of each head
    batch_size = query.size(0)
    
    # Reshape query, key, value for multi-headed attention
    query = query.view(batch_size, -1, num_heads, d_k).transpose(1, 2)  # (batch_size, num_heads, seq_len, d_k)
    key = key.view(batch_size, -1, num_heads, d_k).transpose(1, 2)
    value = value.view(batch_size, -1, num_heads, d_k).transpose(1, 2)
    
    # Scaled dot product attention for each head
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=query.dtype, device=query.device))
    weights = F.softmax(scores, dim=-1)
    output = torch.matmul(weights, value)
    
    # Reshape output to concatenate heads
    output = output.transpose(1, 2).contiguous().view(batch_size, -1, num_heads * d_k)
    return output, weights


class KNeighborsVC(nn.Module):

    def __init__(self,
        wavlm: WavLM,
        hifigan: HiFiGAN,
        hifigan_cfg: AttrDict,
        device='cuda'
    ) -> None:
        """ kNN-VC matcher. 
        Arguments:
            - `wavlm` : trained WavLM model
            - `hifigan`: trained hifigan model
            - `hifigan_cfg`: hifigan config to use for vocoding.
        """
        super().__init__()
        # set which features to extract from wavlm
        self.weighting = torch.tensor(SPEAKER_INFORMATION_WEIGHTS, device=device)[:, None]
        # load hifigan
        self.hifigan = hifigan.eval()
        self.h = hifigan_cfg
        # store wavlm
        self.wavlm = wavlm.eval()
        self.device = torch.device(device)
        self.sr = self.h.sampling_rate
        self.hop_length = 320
        

    def get_matching_set(self, wavs: list[Path] | list[Tensor], weights=None, vad_trigger_level=7) -> Tensor:
        feats = []
        max_length = 0
        
        # First pass: extract features and print shapes
        print("\nFeature extraction shapes:")
        for i, p in enumerate(wavs):
            feat = self.get_features(p, weights=self.weighting if weights is None else weights, 
                                   vad_trigger_level=vad_trigger_level)
            print(f"Wave {i}: Shape = {feat.shape}")
            feats.append(feat)
            max_length = max(max_length, feat.shape[1])  # Time dimension
        
        print(f"\nMaximum length found: {max_length}")
        
        # Second pass: padding with shape verification
        padded_feats = []
        for i, feat in enumerate(feats):
            if feat.shape[1] < max_length:
                print(f"\nPadding Wave {i}:")
                print(f"Original shape: {feat.shape}")
                # Pad the middle dimension (time dimension)
                # The padding tuple specifies padding for each dimension from last to first
                # (0,0) for last dim (features), (0,max_length-feat.shape[1]) for time dim, (0,0) for batch dim
                padded = torch.nn.functional.pad(
                    feat, 
                    pad=(0, 0,  # feature dimension (last)
                         0, max_length - feat.shape[1],  # time dimension (middle)
                         0, 0)  # batch dimension (first)
                )
                print(f"Padded shape: {padded.shape}")
                padded_feats.append(padded)
            else:
                padded_feats.append(feat)
        
        # Verify final shapes before concatenation
        print("\nFinal shapes before concatenation:")
        for i, feat in enumerate(padded_feats):
            print(f"Wave {i}: {feat.shape}")
        
        try:
            # Attempt concatenation
            result = torch.cat(padded_feats, dim=0).cpu()
            print(f"\nFinal concatenated shape: {result.shape}")
            return result
        except RuntimeError as e:
            print("\nError during concatenation!")
            print("Detailed tensor information:")
            for i, feat in enumerate(padded_feats):
                print(f"Wave {i}:")
                print(f"  Shape: {feat.shape}")
                print(f"  Dtype: {feat.dtype}")
                print(f"  Device: {feat.device}")
            raise e

   

    @torch.inference_mode()
    def vocode(self, c: Tensor) -> Tensor:
        """ Vocode features with hifigan. `c` is of shape (bs, seq_len, c_dim) """
        y_g_hat = self.hifigan(c)
        y_g_hat = y_g_hat.squeeze(1)
        return y_g_hat


    @torch.inference_mode()
    def get_features(self, path, weights=None, vad_trigger_level=0):
        """Returns features of `path` waveform as a tensor of shape (seq_len, dim), optionally perform VAD trimming
        on start/end with `vad_trigger_level`.
        """
        # load audio
        if weights == None: weights = self.weighting
        if type(path) in [str, Path]:
            x, sr = torchaudio.load(path, normalize=True)
        else:
            x: Tensor = path
            sr = self.sr
            if x.dim() == 1: x = x[None]
                
        if not sr == self.sr :
            print(f"resample {sr} to {self.sr} in {path}")
            x = torchaudio.functional.resample(x, orig_freq=sr, new_freq=self.sr)
            sr = self.sr
            
        # trim silence from front and back
        if vad_trigger_level > 1e-3:
            transform = T.Vad(sample_rate=sr, trigger_level=vad_trigger_level)
            x_front_trim = transform(x)
            # original way, disabled because it lacks windows support
            #waveform_reversed, sr = apply_effects_tensor(x_front_trim, sr, [["reverse"]])
            waveform_reversed = torch.flip(x_front_trim, (-1,))
            waveform_reversed_front_trim = transform(waveform_reversed)
            waveform_end_trim = torch.flip(waveform_reversed_front_trim, (-1,))
            #waveform_end_trim, sr = apply_effects_tensor(
            #    waveform_reversed_front_trim, sr, [["reverse"]]
            #)
            x = waveform_end_trim

        # extract the representation of each layer
        wav_input_16khz = x.to(self.device)
        if torch.allclose(weights, self.weighting):
            # use fastpath
            features = self.wavlm.extract_features(wav_input_16khz, output_layer=SPEAKER_INFORMATION_LAYER, ret_layer_results=False)[0]
            features = features.squeeze(0)
        else:
            # use slower weighted
            rep, layer_results = self.wavlm.extract_features(wav_input_16khz, output_layer=self.wavlm.cfg.encoder_layers, ret_layer_results=True)[0]
            features = torch.cat([x.transpose(0, 1) for x, _ in layer_results], dim=0) # (n_layers, seq_len, dim)
            # save full sequence
            features = ( features*weights[:, None] ).sum(dim=0) # (seq_len, dim)
        
        return features
    
    

    def get_features_from_directory(self, directory_path, pattern="*.wav", recursive=False):
        """Extract WavLM features from all audio files in a directory and pad them to the same length."""

        from pathlib import Path

        directory = Path(directory_path)
        wav_files = sorted(directory.rglob(pattern) if recursive else directory.glob(pattern))

        if not wav_files:
            raise ValueError(f"No files matching '{pattern}' found in {directory_path}")

        print(f"Found {len(wav_files)} audio files in {directory_path}")

        features_list = []
        wav_paths = [str(f) for f in wav_files]
        max_length = 0  # Track max sequence length

        # Extract features
        for i, wav_path in enumerate(wav_paths):
            try:
                print(f"Processing file {i+1}/{len(wav_paths)}: {wav_path}")
                features = self.get_features(wav_path)  # Shape: [2, T, 1024]
                max_length = max(max_length, features.shape[1])  # Get max time dimension
                features_list.append(features)
            except Exception as e:
                print(f"Error processing {wav_path}: {e}")

        if not features_list:
            raise ValueError("No features could be extracted from the files")

        # Pad all features to max_length
        padded_features = []
        for feat in features_list:
            pad_amount = max_length - feat.shape[1]
            padded_feat = F.pad(feat, (0, 0, 0, pad_amount))  # Pad along time dimension
            padded_features.append(padded_feat)

        return torch.stack(padded_features), wav_paths


    @torch.inference_mode()
#     def match_with_fuzzy_spectral_attention(self, query_seq: torch.Tensor, matching_set: torch.Tensor, 
#                                      synth_set: torch.Tensor = None, tgt_loudness_db: float | None = -16, 
#                                      target_duration: float | None = None, min_clusters: int = 1, 
#                                      max_clusters: int = 10, eigen_solver: str = 'arpack', 
#                                      assign_labels: str = 'kmeans', num_heads: int = 16,
#                                      fcm_m: float = 2.0, fcm_error: float = 1e-5, 
#                                      fcm_maxiter: int = 100, 
#                                      temporal_weight: float = 0.3) -> torch.Tensor:

   

       

#         query_seq, matching_set = query_seq.to(self.device), matching_set.to(self.device)
#         synth_set = matching_set if synth_set is None else synth_set.to(self.device)

#         if target_duration is not None:
#             target_samples = int(target_duration * self.sr)
#             scale_factor = (target_samples / self.hop_length) / query_seq.shape[0]
#             query_seq = torch.nn.functional.interpolate(
#                 query_seq.unsqueeze(0).transpose(1, 2), 
#                 scale_factor=scale_factor, 
#                 mode='linear'
#             ).transpose(1, 2).squeeze(0)

#         query_attention = query_seq.unsqueeze(0)
#         matching_attention = matching_set.unsqueeze(0)
#         synth_attention = synth_set.unsqueeze(0)

#         enhanced_query, _ = multi_head_attention(query_attention, matching_attention, synth_attention, num_heads)
#         enhanced_query = enhanced_query.squeeze(0)

#         query_cpu = enhanced_query.cpu().numpy()
#         matching_cpu = matching_set.cpu().numpy()
#         combined_features = np.vstack([query_cpu, matching_cpu])

#         # SVD for dimensionality reduction
#         n_components = min(combined_features.shape[0], combined_features.shape[1], 50)
#         U, S, Vt = svd(combined_features, full_matrices=False)
#         reduced_features = np.dot(U[:, :n_components], np.diag(S[:n_components]))

#         fcm_objective_values = []
#         K_range = range(min_clusters, max_clusters + 1)

#         for k in K_range:
#             try:
#                 # Use spatial FCM instead of standard FCM
#                 _, _, jm, _ = spatial_fuzzy_c_means(
#                     reduced_features, 
#                     c=k, 
#                     m=fcm_m, 
#                     error=fcm_error, 
#                     maxiter=fcm_maxiter,
#                     lambda_param=temporal_weight
#                 )
#                 fcm_objective_values.append(jm[-1])
#             except Exception:
#                 fcm_objective_values.append(fcm_objective_values[-1] if fcm_objective_values else float('inf'))

#         if len(fcm_objective_values) >= 3:
#             derivatives = np.diff(fcm_objective_values)
#             second_derivatives = np.diff(derivatives)
#             elbow_index = np.argmax(np.abs(second_derivatives)) + 1
#             optimal_k = K_range[elbow_index]
#         else:
#             optimal_k = min_clusters

#         # Use spatial FCM with optimal k
#         fcm_centers, membership_matrix, _, _ = spatial_fuzzy_c_means(
#             reduced_features, 
#             c=optimal_k, 
#             m=fcm_m, 
#             error=fcm_error, 
#             maxiter=fcm_maxiter,
#             lambda_param=temporal_weight
#         )

#         query_memberships = membership_matrix[:len(query_cpu)]
#         matching_memberships = membership_matrix[len(query_cpu):]

#         out_feats_list = []

#         # Add contextual window for matching
#         window_size = 3  # Consider frames before and after current frame

#         for i in range(len(query_cpu)):
#             query_frame_memberships = query_memberships[i]
#             weighted_features = torch.zeros(synth_set.shape[1], device=self.device)
#             total_weight = 0

#             # Consider temporal context
#             context_start = max(0, i - window_size)
#             context_end = min(len(query_cpu), i + window_size + 1)
#             context_weights = np.exp(-0.5 * (np.arange(context_start, context_end) - i)**2 / (window_size/2)**2)
#             context_weights = context_weights / context_weights.sum()

#             # Aggregate membership across temporal window
#             contextual_memberships = np.zeros(optimal_k)
#             for idx, j in enumerate(range(context_start, context_end)):
#                 contextual_memberships += query_memberships[j] * context_weights[idx]

#             for cluster_idx in range(optimal_k):
#                 query_cluster_membership = contextual_memberships[cluster_idx]
#                 if query_cluster_membership < 0.1:
#                     continue

#                 match_indices = np.where(matching_memberships[:, cluster_idx] > 0.1)[0]

#                 if len(match_indices) > 0:
#                     cluster_matches = synth_set[match_indices]
#                     match_memberships = torch.tensor(
#                         matching_memberships[match_indices, cluster_idx], 
#                         dtype=torch.float32, 
#                         device=self.device
#                     ).unsqueeze(1)

#                     query_frame = enhanced_query[i]
#                     query_norm = F.normalize(query_frame.unsqueeze(0), p=2, dim=1)
#                     match_norm = F.normalize(cluster_matches, p=2, dim=1)

#                     # Use cosine similarity with adaptive temperature
#                     similarities = torch.mm(query_norm, match_norm.transpose(0, 1)).squeeze(0)

#                     # Adaptive temperature based on similarity distribution
#                     sim_mean = similarities.mean().item()
#                     sim_std = similarities.std().item() + 1e-6
#                     temperature = 0.1 * (1.0 / (sim_std * min(len(match_indices), 10)))

#                     sim_weights = F.softmax(similarities / temperature, dim=0).unsqueeze(1)

#                     combined_weights = sim_weights * match_memberships
#                     combined_weights = combined_weights / (combined_weights.sum() + 1e-10)

#                     cluster_contribution = torch.sum(cluster_matches * combined_weights, dim=0)
#                     weighted_features += cluster_contribution * query_cluster_membership
#                     total_weight += query_cluster_membership

#             if total_weight > 0:
#                 weighted_features = weighted_features / total_weight
#                 out_feats_list.append(weighted_features)
#             else:
#                 # Enhanced fallback strategy
#                 # Use a weighted combination of nearest neighbors
#                 distances = torch.cdist(enhanced_query[i].unsqueeze(0), matching_set, p=2).squeeze(0)
#                 topk_values, topk_indices = torch.topk(distances, k=5, largest=False)

#                 # Convert distances to weights using softmax
#                 weights = F.softmax(-topk_values / 0.1, dim=0)

#                 # Weighted average of top matches
#                 weighted_match = torch.zeros_like(synth_set[0])
#                 for j, idx in enumerate(topk_indices):
#                     weighted_match += synth_set[idx] * weights[j]

#                 out_feats_list.append(weighted_match)

#         out_feats = torch.stack(out_feats_list)

#         # Apply temporal smoothing
#         if len(out_feats) > 3:
#             # Simple exponential smoothing
#             alpha = 0.7  # Smoothing factor
#             smoothed = out_feats.clone()
#             for i in range(1, len(out_feats)):
#                 smoothed[i] = alpha * out_feats[i] + (1 - alpha) * smoothed[i-1]
#             out_feats = smoothed

#         prediction = self.vocode(out_feats.unsqueeze(0)).cpu().squeeze()

#         if tgt_loudness_db is not None:
#             src_loudness = torchaudio.functional.loudness(prediction.unsqueeze(0), self.sr)
#             pred_wav = torchaudio.functional.gain(prediction, tgt_loudness_db - src_loudness)
#         else:
#             pred_wav = prediction

#         return pred_wav




   
    
    def match_with_fuzzy_spectral_attention(self, query_seq: torch.Tensor, matching_set: torch.Tensor, 
                                         synth_set: torch.Tensor = None, tgt_loudness_db: float | None = -16, 
                                         target_duration: float | None = None, min_clusters: int = 1, 
                                         max_clusters: int = 10, eigen_solver: str = 'arpack', 
                                         assign_labels: str = 'kmeans', num_heads: int = 16,
                                         fcm_m: float = 2.0, fcm_error: float = 1e-5, 
                                         fcm_maxiter: int = 100) -> torch.Tensor:

        import numpy as np
        from scipy.spatial import distance
        from sklearn.metrics.pairwise import cosine_similarity
        from scipy.linalg import svd

        def fuzzy_c_means(data, c, m, error, maxiter):
            n = data.shape[0]
            u = np.random.rand(n, c)
            row_sums = u.sum(axis=1)
            u = u / row_sums[:, np.newaxis]

            jm = []

            for iteration in range(maxiter):
                um = u ** m
                centers = np.dot(um.T, data) / np.sum(um.T, axis=1, keepdims=True)

                d = distance.cdist(data, centers, metric='euclidean')
                d = np.fmax(d, np.finfo(np.float64).eps)

                jm.append(np.sum((d ** 2) * um))

                if iteration > 0 and abs(jm[iteration] - jm[iteration - 1]) < error:
                    break

                d_inv = 1.0 / d
                power = 2.0 / (m - 1)
                d_inv_power = d_inv ** power
                u = d_inv_power / np.sum(d_inv_power, axis=1, keepdims=True)

            return centers, u, jm, d

        query_seq, matching_set = query_seq.to(self.device), matching_set.to(self.device)
        synth_set = matching_set if synth_set is None else synth_set.to(self.device)

        if target_duration is not None:
            target_samples = int(target_duration * self.sr)
            scale_factor = (target_samples / self.hop_length) / query_seq.shape[0]
            query_seq = torch.nn.functional.interpolate(
                query_seq.unsqueeze(0).transpose(1, 2), 
                scale_factor=scale_factor, 
                mode='linear'
            ).transpose(1, 2).squeeze(0)

        query_attention = query_seq.unsqueeze(0)
        matching_attention = matching_set.unsqueeze(0)
        synth_attention = synth_set.unsqueeze(0)

        enhanced_query, _ = multi_head_attention(query_attention, matching_attention, synth_attention, num_heads)
        enhanced_query = enhanced_query.squeeze(0)

        query_cpu = enhanced_query.cpu().numpy()
        matching_cpu = matching_set.cpu().numpy()
        combined_features = np.vstack([query_cpu, matching_cpu])

        # SVD for dimensionality reduction
        n_components = min(combined_features.shape[0], combined_features.shape[1], 50)
        U, S, Vt = svd(combined_features, full_matrices=False)
        reduced_features = np.dot(U[:, :n_components], np.diag(S[:n_components]))

        fcm_objective_values = []
        K_range = range(min_clusters, max_clusters + 1)

        for k in K_range:
            try:
                _, _, jm, _ = fuzzy_c_means(
                    reduced_features, 
                    c=k, 
                    m=fcm_m, 
                    error=fcm_error, 
                    maxiter=fcm_maxiter
                )
                fcm_objective_values.append(jm[-1])
            except Exception:
                fcm_objective_values.append(fcm_objective_values[-1] if fcm_objective_values else float('inf'))

        if len(fcm_objective_values) >= 3:
            derivatives = np.diff(fcm_objective_values)
            second_derivatives = np.diff(derivatives)
            elbow_index = np.argmax(np.abs(second_derivatives)) + 1
            optimal_k = K_range[elbow_index]
        else:
            optimal_k = min_clusters

        fcm_centers, membership_matrix, _, _ = fuzzy_c_means(
            reduced_features, 
            c=optimal_k, 
            m=fcm_m, 
            error=fcm_error, 
            maxiter=fcm_maxiter
        )

        query_memberships = membership_matrix[:len(query_cpu)]
        matching_memberships = membership_matrix[len(query_cpu):]

        out_feats_list = []

        for i in range(len(query_cpu)):
            query_frame_memberships = query_memberships[i]
            weighted_features = torch.zeros(synth_set.shape[1], device=self.device)
            total_weight = 0

            for cluster_idx in range(optimal_k):
                query_cluster_membership = query_frame_memberships[cluster_idx]
                if query_cluster_membership < 0.1:
                    continue

                match_indices = np.where(matching_memberships[:, cluster_idx] > 0.1)[0]

                if len(match_indices) > 0:
                    cluster_matches = synth_set[match_indices]
                    match_memberships = torch.tensor(
                        matching_memberships[match_indices, cluster_idx], 
                        dtype=torch.float32, 
                        device=self.device
                    ).unsqueeze(1)

                    query_frame = enhanced_query[i]
                    query_norm = F.normalize(query_frame.unsqueeze(0), p=2, dim=1)
                    match_norm = F.normalize(cluster_matches, p=2, dim=1)

                    similarities = torch.mm(query_norm, match_norm.transpose(0, 1)).squeeze(0)
                    temperature = 0.1 * (1.0 / min(len(match_indices), 10))
                    sim_weights = F.softmax(similarities / temperature, dim=0).unsqueeze(1)

                    combined_weights = sim_weights * match_memberships
                    combined_weights = combined_weights / combined_weights.sum()

                    cluster_contribution = torch.sum(cluster_matches * combined_weights, dim=0)
                    weighted_features += cluster_contribution * query_cluster_membership
                    total_weight += query_cluster_membership

            if total_weight > 0:
                weighted_features = weighted_features / total_weight
                out_feats_list.append(weighted_features)
            else:
                distances = torch.cdist(enhanced_query[i].unsqueeze(0), matching_set, p=2).squeeze(0)
                _, idx = distances.min(dim=0)
                out_feats_list.append(synth_set[idx])

        out_feats = torch.stack(out_feats_list)
#         out_feats = out_feats.cpu()              # move to CPU if needed
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'

#         # Move the input tensor to the correct device
#         out_feats = out_feats.to(device)

        
      

#         path_feat_LM = "/workspace/Kris/objective_evaluation/jimmy/LM_output_sep/converted_features_alar_eng_0_sep.pt"
#         LM_feat = torch.load(path_feat_LM)
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         LM_feat = LM_feat.to(device).float()

        
        

        
        prediction = self.vocode(out_feats.unsqueeze(0)).cpu().squeeze()

        if tgt_loudness_db is not None:
            src_loudness = torchaudio.functional.loudness(prediction.unsqueeze(0), self.sr)
            pred_wav = torchaudio.functional.gain(prediction, tgt_loudness_db - src_loudness)
        else:
            pred_wav = prediction

        return pred_wav

















#### no use ######
#     def match_with_fuzzy_spectral_attention(self, query_seq: torch.Tensor, matching_set: torch.Tensor, 
#                                      synth_set: torch.Tensor = None, tgt_loudness_db: float | None = -16, 
#                                      target_duration: float | None = None, min_clusters: int = 1, 
#                                      max_clusters: int = 10, eigen_solver: str = 'arpack', 
#                                      assign_labels: str = 'kmeans', num_heads: int = 16,
#                                      fcm_m: float = 2.0, fcm_error: float = 1e-5, 
#                                      fcm_maxiter: int = 100) -> torch.Tensor:
   
  
    
#     # Define Fuzzy C-Means clustering implementation
#         def fuzzy_c_means(data, c, m, error, maxiter):
#             """
#             Fuzzy C-Means clustering algorithm

#             Args:
#                 data: Input data matrix (n_samples, n_features)
#                 c: Number of clusters
#                 m: Fuzziness parameter (m > 1)
#                 error: Error threshold for convergence
#                 maxiter: Maximum number of iterations

#             Returns:
#                 centers: Cluster centers
#                 u: Membership matrix (n_samples, c)
#                 jm: Objective function history
#                 d: Final distances from points to cluster centers
#             """
#             n = data.shape[0]  # number of samples

#             # Initialize membership matrix with random values
#             u = np.random.rand(n, c)
#             # Normalize rows to sum to 1
#             row_sums = u.sum(axis=1)
#             u = u / row_sums[:, np.newaxis]

#             jm = []  # objective function history

#             for iteration in range(maxiter):
#                 # Calculate cluster centers
#                 um = u ** m
#                 centers = np.dot(um.T, data) / np.sum(um.T, axis=1, keepdims=True)

#                 # Calculate distances
#                 d = distance.cdist(data, centers, metric='euclidean')
#                 d = np.fmax(d, np.finfo(np.float64).eps)  # Avoid division by zero

#                 # Update objective function
#                 jm.append(np.sum((d ** 2) * um))

#                 # Check for convergence
#                 if iteration > 0 and abs(jm[iteration] - jm[iteration-1]) < error:
#                     break

#                 # Update membership matrix
#                 d_inv = 1.0 / d
#                 power = 2.0 / (m - 1)
#                 d_inv_power = d_inv ** power
#                 u = d_inv_power / np.sum(d_inv_power, axis=1, keepdims=True)

#             return centers, u, jm, d
    
#         query_seq, matching_set = query_seq.to(self.device), matching_set.to(self.device)
#         if synth_set is None:
#             synth_set = matching_set
#         else:
#             synth_set = synth_set.to(self.device)

#         # Print shapes for debugging
#         print(f"Query sequence shape: {query_seq.shape}")
#         print(f"Matching set shape: {matching_set.shape}")
#         if synth_set is not None:
#             print(f"Synthesis set shape: {synth_set.shape}")

#         # Time interpolation if target duration is specified
#         if target_duration is not None:
#             target_samples = int(target_duration * self.sr)
#             scale_factor = (target_samples / self.hop_length) / query_seq.shape[0]
#             query_seq = torch.nn.functional.interpolate(
#                 query_seq.unsqueeze(0).transpose(1, 2), 
#                 scale_factor=scale_factor, 
#                 mode='linear'
#             ).transpose(1, 2).squeeze(0)

#         # Apply multi-head attention to enhance feature representation
#         query_attention = query_seq.unsqueeze(0)  # Add batch dimension
#         matching_attention = matching_set.unsqueeze(0)
#         synth_attention = synth_set.unsqueeze(0)

#         # Apply self-attention on query sequence
#         enhanced_query, _ = multi_head_attention(query_attention, matching_attention, synth_attention, num_heads)
#         enhanced_query = enhanced_query.squeeze(0)  # Remove batch dimension

#         # Move tensors to CPU for scikit-learn and numpy compatibility
#         query_cpu = enhanced_query.cpu().numpy()
#         matching_cpu = matching_set.cpu().numpy()

#         # Prepare features for clustering
#         combined_features = np.vstack([query_cpu, matching_cpu])

#         # Create similarity matrix (using cosine similarity)
#         similarity_matrix = cosine_similarity(combined_features)

#         # Apply PCA for dimensionality reduction
#         pca = PCA(n_components=min(combined_features.shape[0], combined_features.shape[1], 50))
#         reduced_features = pca.fit_transform(combined_features)

#         # Use elbow method with Fuzzy C-Means to determine optimal cluster number
#         fcm_objective_values = []
#         K_range = range(min_clusters, max_clusters + 1)

#         for k in K_range:
#             try:
#                 # Apply Fuzzy C-Means clustering
#                 _, _, jm, _ = fuzzy_c_means(
#                     reduced_features, 
#                     c=k, 
#                     m=fcm_m, 
#                     error=fcm_error, 
#                     maxiter=fcm_maxiter
#                 )
#                 # Store final objective function value
#                 fcm_objective_values.append(jm[-1])
#                 print(f"Clusters: {k}, FCM Objective: {jm[-1]:.4f}")
#             except Exception as e:
#                 print(f"Error with cluster count {k}: {str(e)}")
#                 # Add a placeholder value if clustering fails
#                 if len(fcm_objective_values) > 0:
#                     fcm_objective_values.append(fcm_objective_values[-1])
#                 else:
#                     fcm_objective_values.append(float('inf'))

#         # Calculate derivatives to find elbow point
#         if len(fcm_objective_values) >= 3:
#             derivatives = np.diff(fcm_objective_values)
#             second_derivatives = np.diff(derivatives)

#             # Find elbow point where second derivative is maximized
#             elbow_index = np.argmax(np.abs(second_derivatives)) + 1
#             optimal_k = K_range[elbow_index]
#         else:
#             # Fallback if we don't have enough points
#             optimal_k = min_clusters

#         print(f"FCM Objective values: {fcm_objective_values}")
#         print(f"Selected optimal number of clusters using elbow method: {optimal_k}")

#         # Apply Fuzzy C-Means with optimal cluster number
#         fcm_centers, membership_matrix, _, _ = fuzzy_c_means(
#             reduced_features, 
#             c=optimal_k, 
#             m=fcm_m, 
#             error=fcm_error, 
#             maxiter=fcm_maxiter
#         )

#         # Separate membership matrices for query and matching
#         query_memberships = membership_matrix[:len(query_cpu)]
#         matching_memberships = membership_matrix[len(query_cpu):]

#         # Process each frame in the query sequence using fuzzy memberships
#         out_feats_list = []

#         for i in range(len(query_cpu)):
#             # Get cluster memberships for this query frame
#             query_frame_memberships = query_memberships[i]

#             # Initialize weighted feature accumulator
#             weighted_features = torch.zeros(synth_set.shape[1], device=self.device)
#             total_weight = 0

#             # For each cluster, find matching frames with significant membership
#             for cluster_idx in range(optimal_k):
#                 # Get query frame's membership to this cluster
#                 query_cluster_membership = query_frame_memberships[cluster_idx]

#                 # Skip if membership is insignificant
#                 if query_cluster_membership < 0.1:  # Threshold can be adjusted
#                     continue

#                 # Find matching frames with high membership to this cluster
#                 # Get indices of frames with significant membership to this cluster
#                 match_indices = np.where(matching_memberships[:, cluster_idx] > 0.1)[0]

#                 if len(match_indices) > 0:
#                     # Get features and memberships
#                     cluster_matches = synth_set[match_indices]
#                     match_memberships = torch.tensor(
#                         matching_memberships[match_indices, cluster_idx], 
#                         dtype=torch.float32, 
#                         device=self.device
#                     ).unsqueeze(1)

#                     # Calculate similarity between query and matches
#                     query_frame = enhanced_query[i]
#                     query_norm = F.normalize(query_frame.unsqueeze(0), p=2, dim=1)
#                     match_norm = F.normalize(cluster_matches, p=2, dim=1)

#                     # Compute similarities
#                     similarities = torch.mm(query_norm, match_norm.transpose(0, 1)).squeeze(0)

#                     # Combine FCM memberships with similarities
#                     # Adjust temperature based on cluster size
#                     temperature = 0.1 * (1.0 / min(len(match_indices), 10))
#                     sim_weights = F.softmax(similarities / temperature, dim=0).unsqueeze(1)

#                     # Combine membership weights with similarity weights
#                     combined_weights = sim_weights * match_memberships
#                     combined_weights = combined_weights / combined_weights.sum()

#                     # Apply weights to get weighted average for this cluster
#                     cluster_contribution = torch.sum(cluster_matches * combined_weights, dim=0)

#                     # Add to total weighted features based on query's membership to this cluster
#                     weighted_features += cluster_contribution * query_cluster_membership
#                     total_weight += query_cluster_membership

#             # Handle case where no significant memberships were found
#             if total_weight > 0:
#                 # Normalize by total weight
#                 weighted_features = weighted_features / total_weight
#                 out_feats_list.append(weighted_features)
#             else:
#                 # Fallback: find closest match using Euclidean distance
#                 distances = torch.cdist(enhanced_query[i].unsqueeze(0), matching_set, p=2).squeeze(0)
#                 _, idx = distances.min(dim=0)
#                 out_feats_list.append(synth_set[idx])

#         # Stack the results
#         out_feats = torch.stack(out_feats_list)

#         # Save the output features if needed
# #         file_name = 'transformed_features-2.pt'
# #         path = "/workspace/Kris/objective_evaluation/enhanced_feats_with_DTW/"
# #         save_path = os.path.join(path, file_name)
# #         torch.save(out_feats, save_path)
        
#         path_feat_LM = "/workspace/Kris/knn-vc/English_Speech_Feat/DTW_augmented/transformed_segment_005_002.pt"
#         LM_feat = torch.load(path_feat_LM)
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         LM_feat = LM_feat.to(device).float()

#         # Vocode the features
#         prediction = self.vocode(out_feats.unsqueeze(0)).cpu().squeeze()

#         # Normalize if needed
#         if tgt_loudness_db is not None:
#             src_loudness = torchaudio.functional.loudness(prediction.unsqueeze(0), self.sr)
#             tgt_loudness = tgt_loudness_db
#             pred_wav = torchaudio.functional.gain(prediction, tgt_loudness - src_loudness)
#         else:
#             pred_wav = prediction

#         return pred_wav
































  
  

#     def match(self, query_seq: torch.Tensor, matching_set: torch.Tensor, synth_set: torch.Tensor = None,
#               tgt_loudness_db: float | None = -16, target_duration: float | None = None, p_norm: int = 2, radius: float = 0.5, num_heads=16) -> torch.Tensor:
#         query_seq, matching_set = query_seq.to(self.device), matching_set.to(self.device)
#         if synth_set is None:
#             synth_set = matching_set
#         else:
#             synth_set = synth_set.to(self.device)
    
#         if target_duration is not None:
#             target_samples = int(target_duration * self.sr)
#             scale_factor = (target_samples / self.hop_length) / query_seq.shape[0]
#             query_seq = torch.nn.functional.interpolate(query_seq.unsqueeze(0).transpose(1, 2), scale_factor=scale_factor, mode='linear').transpose(1, 2).squeeze(0)
    
#         # Applying self-attention
#         attention_output, _ = multi_head_attention(query_seq.unsqueeze(0), matching_set.unsqueeze(0), synth_set.unsqueeze(0), num_heads)
#         out_feats = attention_output.squeeze(0)
    
#             # Calculate distances
#         distances = torch.cdist(out_feats.unsqueeze(0), matching_set, p=p_norm).squeeze(0)
            
#             # Determine dynamic k based on local density or distance thresholds
#         local_density_counts = (distances < radius).sum(1)
#         dynamic_k = torch.clamp(local_density_counts, min=1, max=len(matching_set))  # Ensure k is at least 1 and not more than the number of samples
        
#         # For each query, find top dynamic_k nearest neighbors
#         out_feats_list = []
#         for i, k in enumerate(dynamic_k):
#             _, indices = distances[i].topk(k=k, largest=False)
#             out_feats_list.append(synth_set[indices].mean(dim=0))
    
#         # Average the results from dynamically chosen neighbors
#         out_feats = torch.stack(out_feats_list)
#         file_name = 'output_feat_transInput.pt'
#         print("out_feats", out_feats.shape)
#         path = "/workspace/Kris/knn-vc/"
#         save_path = os.path.join(path, file_name)

        
#         torch.save(out_feats, save_path)
#         path_feat_LM = "/workspace/Kris/objective_evaluation/jimmy/LM_output_sep/converted_features_alar_eng_0_sep.pt"
#         LM_feat = torch.load(path_feat_LM)
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         LM_feat = LM_feat.to(device).float()
    
        

    
#         prediction = self.vocode(out_feats.unsqueeze(0)).cpu().squeeze()  # Placeholder for your vocode method
    
#         # Normalization
#         if tgt_loudness_db is not None:
#             src_loudness = torchaudio.functional.loudness(prediction.unsqueeze(0), self.sr)
#             tgt_loudness = tgt_loudness_db
#             pred_wav = torchaudio.functional.gain(prediction, tgt_loudness - src_loudness)
#         else:
#             pred_wav = prediction
    
#         return pred_wav
    
    

    