import numpy as np
import soundfile as sf
import librosa
from pesq import pesq

def evaluate_pesq(clean_path, noisy_path, denoised_path, sr=16000):
    """
    Evaluate PESQ (Perceptual Evaluation of Speech Quality) for Noisy and Denoised speech.
    
    Args:
        clean_path (str): Path to clean audio file
        noisy_path (str): Path to noisy audio file
        denoised_path (str): Path to denoised audio file
        sr (int): Target sample rate (PESQ supports only 8000 or 16000 Hz)
    
    Returns:
        None (prints PESQ scores)
    """
    # Load audio files and resample if needed
    clean_signal, sr_clean = librosa.load(clean_path, sr=sr)
    noisy_signal, sr_noisy = librosa.load(noisy_path, sr=sr)
    denoised_signal, sr_denoised = librosa.load(denoised_path, sr=sr)

    # Compute PESQ
    pesq_noisy = pesq(sr, clean_signal, noisy_signal, 'wb')  # Wideband PESQ
    pesq_denoised = pesq(sr, clean_signal, denoised_signal, 'wb')

    print("=== PESQ Evaluation ===")
    print(f"  PESQ (Noisy): {pesq_noisy:.3f}")
    print(f"  PESQ (Denoised): {pesq_denoised:.3f}")
    print(f"  Improvement: {pesq_denoised - pesq_noisy:.3f}")

# Example usage
if __name__ == "__main__":
    clean_path = "/workspace/Kris/knn-vc/Denoising_Work/test_files/p232_005_c.wav"
    noisy_path = "/workspace/Kris/knn-vc/Denoising_Work/test_files/p232_005_n.wav"
    denoised_path = "/workspace/Kris/knn-vc/Denoising_Work/BigVGAN/output.wav"

    evaluate_pesq(clean_path, noisy_path, denoised_path)
