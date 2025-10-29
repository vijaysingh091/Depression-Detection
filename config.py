"""
config.py - Central configuration file for depression detection project
"""

from pathlib import Path

class Config:
    """Configuration class for the entire project"""
    
    # ============ PATHS ============
    PROJECT_ROOT = Path(__file__).parent
    DATA_ROOT = PROJECT_ROOT / 'data'
    
    RAW_DATA = DATA_ROOT / 'raw' / 'DAIC-WOZ'
    PROCESSED_DATA = DATA_ROOT / 'processed'
    FEATURES_DATA = DATA_ROOT / 'features'
    
    MODELS_DIR = PROJECT_ROOT / 'models'
    SAVED_MODELS = MODELS_DIR / 'saved_models'
    CHECKPOINTS = MODELS_DIR / 'checkpoints'
    
    RESULTS_DIR = PROJECT_ROOT / 'results'
    FIGURES_DIR = RESULTS_DIR / 'figures'
    METRICS_DIR = RESULTS_DIR / 'metrics'
    ATTENTION_MAPS_DIR = RESULTS_DIR / 'attention_maps'
    
    LOGS_DIR = PROJECT_ROOT / 'logs'
    
    TRAIN_SPLIT_CSV = RAW_DATA / 'train_split_Depression_AVEC2017.csv'
    DEV_SPLIT_CSV = RAW_DATA / 'dev_split_Depression_AVEC2017.csv'
    TEST_SPLIT_CSV = RAW_DATA / 'test_split_Depression_AVEC2017.csv'
    
    # ============ DATA PARAMETERS ============
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_DURATION_MAX = 600
    N_MFCC = 13
    HOP_LENGTH = 512
    N_FFT = 2048
    
    VIDEO_FPS = 30
    FACE_SIZE = (224, 224)
    N_FAU = 17
    
    MAX_SEQUENCE_LENGTH = 512
    BERT_MODEL = 'bert-base-uncased'
    BERT_HIDDEN_SIZE = 768
    
    # ============ MODEL PARAMETERS ============
    RANDOM_SEED = 42
    DEVICE = 'cpu'  # Change to 'cuda' if you have GPU
    
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    PATIENCE = 10
    MIN_DELTA = 0.001
    
    HIDDEN_SIZE = 256
    NUM_LAYERS = 2
    DROPOUT = 0.3
    
    ATTENTION_HEADS = 8
    ATTENTION_DROPOUT = 0.1
    
    # ============ TARGET VARIABLES ============
    PHQ8_MIN = 0
    PHQ8_MAX = 24
    
    SEVERITY_THRESHOLDS = {
        'None': (0, 4),
        'Mild': (5, 9),
        'Moderate': (10, 14),
        'Severe': (15, 24)
    }
    
    NUM_CLASSES = 4
    
    AUDIO_FEATURE_DIM = 39
    VIDEO_FEATURE_DIM = 17
    TEXT_FEATURE_DIM = 768
    
    METRICS = ['mae', 'mse', 'rmse', 'r2', 'f1', 'accuracy', 'precision', 'recall']
    
    LOG_INTERVAL = 10
    SAVE_INTERVAL = 5
    
    PLOT_DPI = 300
    FIGURE_SIZE = (12, 8)
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories"""
        directories = [
            cls.DATA_ROOT, cls.RAW_DATA, cls.PROCESSED_DATA,
            cls.FEATURES_DATA, cls.MODELS_DIR, cls.SAVED_MODELS,
            cls.CHECKPOINTS, cls.RESULTS_DIR, cls.FIGURES_DIR,
            cls.METRICS_DIR, cls.ATTENTION_MAPS_DIR, cls.LOGS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        print("✓ All directories created")
    
    @classmethod
    def get_session_paths(cls, session_id):
        """Get all file paths for a given session"""
        session_dir = cls.RAW_DATA / f"{session_id}_P"
        
        return {
            'audio': session_dir / f"{session_id}_AUDIO.wav",
            'video': session_dir / f"{session_id}_VIDEO.mp4",
            'transcript': session_dir / f"{session_id}_TRANSCRIPT.csv",
            'openface': session_dir / f"{session_id}_OpenFace.csv"
        }
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("="*60)
        print("PROJECT CONFIGURATION")
        print("="*60)
        print(f"\nPaths:")
        print(f"  Project Root: {cls.PROJECT_ROOT}")
        print(f"  Raw Data: {cls.RAW_DATA}")
        print(f"  Models: {cls.MODELS_DIR}")
        print(f"\nData Parameters:")
        print(f"  Audio Sample Rate: {cls.AUDIO_SAMPLE_RATE} Hz")
        print(f"  BERT Model: {cls.BERT_MODEL}")
        print(f"\nModel Parameters:")
        print(f"  Batch Size: {cls.BATCH_SIZE}")
        print(f"  Learning Rate: {cls.LEARNING_RATE}")
        print(f"  Epochs: {cls.NUM_EPOCHS}")
        print("="*60)


def get_config():
    return Config()


if __name__ == "__main__":
    Config.print_config()
    Config.create_directories()