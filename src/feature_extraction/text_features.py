"""
Text Feature Extraction Script
Extracts BERT embeddings and sentiment from interview transcripts

Place this file in: src/feature_extraction/text_features.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class TextFeatureExtractor:
    """Extract text features using BERT"""
    
    def __init__(self, model_name='bert-base-uncased'):
        """
        Initialize BERT model and tokenizer
        
        Args:
            model_name: Which BERT model to use
        """
        print("Loading BERT model (this may take a minute on first run)...")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode
        
        # Use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        print(f"✓ BERT loaded on device: {self.device}")
        
    def load_transcript(self, transcript_path):
        """
        Load interview transcript
        
        Transcripts are usually CSV files with columns like:
        - speaker: Participant or Ellie (interviewer)
        - text: What was said
        - start_time, end_time: Timing
        
        Args:
            transcript_path: Path to transcript file
            
        Returns:
            DataFrame with transcript
        """
        try:
            # Try different possible formats
            if transcript_path.suffix == '.csv':
                df = pd.read_csv(transcript_path)
            elif transcript_path.suffix == '.txt':
                # Plain text file
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                df = pd.DataFrame({'text': [text]})
            else:
                print(f"⚠ Unknown transcript format: {transcript_path.suffix}")
                return None
            
            print(f"✓ Loaded transcript: {transcript_path.name}")
            print(f"  Rows: {len(df)}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading transcript: {e}")
            return None
    
    def get_participant_text(self, transcript_df):
        """
        Extract only the participant's speech (not the interviewer)
        
        Args:
            transcript_df: Transcript dataframe
            
        Returns:
            Combined text from participant only
        """
        # Look for common column names for speaker identification
        speaker_column = None
        text_column = None
        
        # Find speaker column
        for col in transcript_df.columns:
            col_lower = col.lower()
            if 'speaker' in col_lower or 'who' in col_lower:
                speaker_column = col
                break
        
        # Find text column
        for col in transcript_df.columns:
            col_lower = col.lower()
            if 'text' in col_lower or 'value' in col_lower or 'transcript' in col_lower:
                text_column = col
                break
        
        if speaker_column and text_column:
            # Filter for participant only (not Ellie/interviewer)
            participant_rows = transcript_df[
                transcript_df[speaker_column].str.contains('Participant', case=False, na=False)
            ]
            
            if len(participant_rows) > 0:
                combined_text = ' '.join(participant_rows[text_column].astype(str).tolist())
                print(f"✓ Extracted participant speech: {len(participant_rows)} utterances")
                return combined_text
        
        # Fallback: use all text
        if text_column:
            combined_text = ' '.join(transcript_df[text_column].astype(str).tolist())
            print(f"⚠ Using all text (couldn't separate speaker)")
            return combined_text
        
        print(f"❌ Could not extract text from transcript")
        return ""
    
    def get_bert_embedding(self, text, max_length=512):
        """
        Get BERT embedding for text
        
        BERT converts text into a 768-dimensional vector that captures meaning.
        
        Args:
            text: Input text
            max_length: Maximum tokens (BERT limit is 512)
            
        Returns:
            768-dimensional embedding vector
        """
        if not text or len(text) == 0:
            return np.zeros(768)  # Return zero vector if no text
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=max_length,
            truncation=True,
            padding='max_length'
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get BERT output
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use [CLS] token embedding (first token)
        # This is a common way to get a single vector for the whole text
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        
        return embedding
    
    def extract_sentiment_features(self, text):
        """
        Extract simple sentiment features
        
        Count positive/negative words, question marks, etc.
        
        Returns:
            Dictionary with sentiment features
        """
        # Simple sentiment word lists
        positive_words = ['good', 'great', 'happy', 'better', 'well', 'fine', 'nice', 'wonderful']
        negative_words = ['bad', 'sad', 'worse', 'terrible', 'awful', 'depressed', 'anxious', 'worried']
        
        text_lower = text.lower()
        words = text_lower.split()
        
        features = {
            'word_count': len(words),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'positive_word_count': sum(1 for w in words if w in positive_words),
            'negative_word_count': sum(1 for w in words if w in negative_words),
            'question_count': text.count('?'),
            'exclamation_count': text.count('!'),
        }
        
        # Sentiment ratio
        total_sentiment = features['positive_word_count'] + features['negative_word_count']
        if total_sentiment > 0:
            features['positive_ratio'] = features['positive_word_count'] / total_sentiment
        else:
            features['positive_ratio'] = 0.5  # Neutral
        
        return features
    
    def extract_all_features(self, transcript_path):
        """
        Extract ALL text features from a transcript
        
        Args:
            transcript_path: Path to transcript file
            
        Returns:
            Dictionary with all features
        """
        print(f"\n{'='*60}")
        print(f"Processing: {Path(transcript_path).name}")
        print(f"{'='*60}")
        
        # Load transcript
        transcript_df = self.load_transcript(transcript_path)
        if transcript_df is None:
            return None
        
        # Get participant text
        text = self.get_participant_text(transcript_df)
        
        if not text:
            print("❌ No text extracted")
            return None
        
        print(f"✓ Text length: {len(text)} characters")
        
        # Get BERT embedding
        print("Extracting BERT embedding...")
        embedding = self.get_bert_embedding(text)
        
        # Get sentiment features
        sentiment_features = self.extract_sentiment_features(text)
        
        # Combine everything
        all_features = {}
        
        # Add BERT embedding (768 dimensions)
        for i in range(len(embedding)):
            all_features[f'bert_dim_{i}'] = embedding[i]
        
        # Add sentiment features
        all_features.update(sentiment_features)
        
        print(f"✓ Total features: {len(all_features)}")
        print(f"{'='*60}\n")
        
        return all_features


def process_all_sessions(data_dir, output_file):
    """
    Process all transcript files in the dataset
    
    Args:
        data_dir: Directory containing session folders
        output_file: Where to save features
    """
    extractor = TextFeatureExtractor()
    
    data_dir = Path(data_dir)
    all_features = []
    
    # Find all session directories
    session_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.endswith('_P')])
    
    print(f"\nFound {len(session_dirs)} sessions to process\n")
    
    for session_dir in tqdm(session_dirs, desc="Processing transcripts"):
        # Get session ID
        session_id = session_dir.name.replace('_P', '')
        
        # Find transcript file
        transcript_files = list(session_dir.glob(f"{session_id}_TRANSCRIPT.*"))
        
        if not transcript_files:
            print(f"⚠ No transcript for session {session_id}")
            continue
        
        transcript_path = transcript_files[0]
        
        # Extract features
        features = extractor.extract_all_features(transcript_path)
        
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
    extractor = TextFeatureExtractor()
    
    # UPDATE THIS PATH
    test_transcript = r"C:\Users\VIJAY BHUSHAN SINGH\depression_detection_project\data\raw\DAIC-WOZ\300_P\300_TRANSCRIPT.csv"
    
    features = extractor.extract_all_features(test_transcript)
    
    if features:
        print("\nSample features:")
        # Show first few BERT dimensions
        for i in range(5):
            print(f"  bert_dim_{i}: {features[f'bert_dim_{i}']:.4f}")
        # Show sentiment features
        print(f"  word_count: {features['word_count']}")
        print(f"  positive_ratio: {features['positive_ratio']:.3f}")
    
    # To process all sessions:
    # data_dir = r"C:\Users\VIJAY BHUSHAN SINGH\depression_detection_project\data\raw\DAIC-WOZ"
    # output_file = r"C:\Users\VIJAY BHUSHAN SINGH\depression_detection_project\data\features\text_features.csv"
    # df = process_all_sessions(data_dir, output_file)