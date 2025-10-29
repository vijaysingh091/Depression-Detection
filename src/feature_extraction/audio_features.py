"""
Audio Feature Extraction Script
Extracts MFCCs, pitch, jitter, shimmer, and energy from audio files

Place this file in: src/feature_extraction/audio_features.py
"""

import librosa
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class AudioFeatureExtractor:
    """Extract audio features for depression detection"""
    
    def __init__(self, sample_rate=16000):
        """
        Initialize the audio feature extractor
        
        Args:
            sample_rate: Target sample rate in Hz (16000 is standard for speech)
        """
        self.sample_rate = sample_rate
        self.n_mfcc = 13  # Number of MFCC coefficients
        
    def load_audio(self, audio_path):
        """
        Load an audio file
        
        Args:
            audio_path: Path to WAV file
            
        Returns:
            audio: Audio signal as numpy array
            sr: Sample rate
        """
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            print(f"✓ Loaded: {Path(audio_path).name}")
            print(f"  Duration: {len(audio)/sr:.2f} seconds")
            return audio, sr
        except Exception as e:
            print(f"❌ Error loading {audio_path}: {e}")
            return None, None
    
    def extract_mfcc(self, audio, sr):
        """
        Extract MFCC features
        
        MFCCs capture the "shape" of the vocal tract - how the voice sounds.
        We get 13 coefficients that describe different aspects of the sound.
        
        Returns:
            Dictionary with MFCC statistics
        """
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
        
        # Calculate statistics over time
        # We want: mean (average), std (variation), min, max
        features = {}
        
        for i in range(self.n_mfcc):
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_std'] = np.std(mfccs[i])
            features[f'mfcc_{i}_min'] = np.min(mfccs[i])
            features[f'mfcc_{i}_max'] = np.max(mfccs[i])
        
        print(f"✓ Extracted {self.n_mfcc} MFCCs")
        return features
    
    def extract_pitch(self, audio, sr):
        """
        Extract pitch (F0) features
        
        Pitch = how high or low the voice is
        Depressed people often have:
        - Lower average pitch
        - Less pitch variation (monotone)
        
        Returns:
            Dictionary with pitch statistics
        """
        # Extract pitch using piptrack
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        
        # Get the most prominent pitch at each time frame
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:  # Only keep valid pitches
                pitch_values.append(pitch)
        
        if len(pitch_values) == 0:
            # No pitch detected (silence or noise)
            return {
                'pitch_mean': 0,
                'pitch_std': 0,
                'pitch_min': 0,
                'pitch_max': 0,
                'pitch_range': 0
            }
        
        pitch_values = np.array(pitch_values)
        
        features = {
            'pitch_mean': np.mean(pitch_values),
            'pitch_std': np.std(pitch_values),
            'pitch_min': np.min(pitch_values),
            'pitch_max': np.max(pitch_values),
            'pitch_range': np.max(pitch_values) - np.min(pitch_values)
        }
        
        print(f"✓ Extracted pitch (mean: {features['pitch_mean']:.1f} Hz)")
        return features
    
    def extract_energy(self, audio):
        """
        Extract energy/intensity features
        
        Energy = how loud the speech is
        Depressed people often speak more quietly
        
        Returns:
            Dictionary with energy statistics
        """
        # RMS (Root Mean Square) energy
        rms = librosa.feature.rms(y=audio)[0]
        
        features = {
            'energy_mean': np.mean(rms),
            'energy_std': np.std(rms),
            'energy_min': np.min(rms),
            'energy_max': np.max(rms)
        }
        
        print(f"✓ Extracted energy features")
        return features
    
    def extract_spectral_features(self, audio, sr):
        """
        Extract spectral features
        
        Spectral features describe the frequency content of speech
        
        Returns:
            Dictionary with spectral features
        """
        # Spectral centroid: "center of mass" of the spectrum
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        
        # Zero crossing rate: how often the signal changes sign
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        
        features = {
            'spectral_centroid_mean': np.mean(spectral_centroids),
            'spectral_centroid_std': np.std(spectral_centroids),
            'zcr_mean': np.mean(zcr),
            'zcr_std': np.std(zcr)
        }
        
        print(f"✓ Extracted spectral features")
        return features
    
    def extract_all_features(self, audio_path):
        """
        Extract ALL audio features from a file
        
        This is the main function you'll call!
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with all features
        """
        print(f"\n{'='*60}")
        print(f"Processing: {Path(audio_path).name}")
        print(f"{'='*60}")
        
        # Load audio
        audio, sr = self.load_audio(audio_path)
        if audio is None:
            return None
        
        # Extract all feature types
        all_features = {}
        
        # MFCCs
        mfcc_features = self.extract_mfcc(audio, sr)
        all_features.update(mfcc_features)
        
        # Pitch
        pitch_features = self.extract_pitch(audio, sr)
        all_features.update(pitch_features)
        
        # Energy
        energy_features = self.extract_energy(audio)
        all_features.update(energy_features)
        
        # Spectral
        spectral_features = self.extract_spectral_features(audio, sr)
        all_features.update(spectral_features)
        
        print(f"\n✓ Total features extracted: {len(all_features)}")
        print(f"{'='*60}\n")
        
        return all_features


def process_all_sessions(data_dir, output_file):
    """
    Process all audio files in the dataset
    
    Args:
        data_dir: Directory containing session folders (e.g., 300_P, 301_P, ...)
        output_file: Where to save the features (CSV file)
    """
    extractor = AudioFeatureExtractor()
    
    data_dir = Path(data_dir)
    all_features = []
    
    # Find all session directories
    session_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.endswith('_P')])
    
    print(f"Found {len(session_dirs)} sessions to process\n")
    
    for session_dir in session_dirs:
        # Get session ID (e.g., 300 from "300_P")
        session_id = session_dir.name.replace('_P', '')
        
        # Find audio file (could be .wav or other format)
        audio_files = list(session_dir.glob(f"{session_id}_AUDIO.*"))
        
        if not audio_files:
            print(f"⚠ No audio file found for session {session_id}")
            continue
        
        audio_path = audio_files[0]
        
        # Extract features
        features = extractor.extract_all_features(audio_path)
        
        if features:
            features['session_id'] = session_id
            all_features.append(features)
    
    # Save to CSV
    df = pd.DataFrame(all_features)
    df.to_csv(output_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"✓ Processed {len(all_features)} sessions")
    print(f"✓ Features saved to: {output_file}")
    print(f"✓ Feature matrix shape: {df.shape}")
    print(f"{'='*60}")
    
    return df


# Example usage
if __name__ == "__main__":
    # Test on a single file
    extractor = AudioFeatureExtractor()
    
    # UPDATE THIS PATH to your actual audio file
    test_audio = r"C:\Users\VIJAY BHUSHAN SINGH\depression_detection_project\data\raw\DAIC-WOZ\300_P\300_AUDIO.wav"
    
    features = extractor.extract_all_features(test_audio)
    
    if features:
        print("\nSample features:")
        for key, value in list(features.items())[:10]:  # Show first 10
            print(f"  {key}: {value:.4f}")
    
    # To process all sessions:
    # data_dir = r"C:\Users\VIJAY BHUSHAN SINGH\depression_detection_project\data\raw\DAIC-WOZ"
    # output_file = r"C:\Users\VIJAY BHUSHAN SINGH\depression_detection_project\data\features\audio_features.csv"
    # df = process_all_sessions(data_dir, output_file)