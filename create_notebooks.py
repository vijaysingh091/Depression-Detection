"""
Create starter Jupyter notebooks for Windows
"""

import os
import json

def create_notebook(filename, cells):
    """Create a Jupyter notebook file"""
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"✓ Created: {filename}")

# Notebook 1: Environment Test
env_test_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["# Environment Setup Test\nRun all cells to verify installation."]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "print(f'Python version: {sys.version}')\n",
            "\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "import torch\n",
            "import librosa\n",
            "import cv2\n",
            "from transformers import BertTokenizer\n",
            "\n",
            "print('\\n✓ All imports successful!')\n",
            "print(f'NumPy: {np.__version__}')\n",
            "print(f'Pandas: {pd.__version__}')\n",
            "print(f'PyTorch: {torch.__version__}')\n",
            "print(f'CUDA available: {torch.cuda.is_available()}')"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Test plotting\n",
            "plt.figure(figsize=(10, 4))\n",
            "x = np.linspace(0, 10, 100)\n",
            "plt.plot(x, np.sin(x), label='sin(x)')\n",
            "plt.plot(x, np.cos(x), label='cos(x)')\n",
            "plt.legend()\n",
            "plt.title('Matplotlib Test')\n",
            "plt.show()\n",
            "print('✓ Plotting works!')"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Test PyTorch\n",
            "x = torch.randn(3, 3)\n",
            "print('Random tensor:')\n",
            "print(x)\n",
            "print(f'\\nDevice: {x.device}')\n",
            "print('✓ PyTorch works!')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## ✓ All Tests Passed!\nYour environment is ready."]
    }
]

# Notebook 2: Data Exploration  
data_exploration_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["# DAIC-WOZ Dataset Exploration"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from pathlib import Path\n",
            "\n",
            "sns.set_style('whitegrid')\n",
            "plt.rcParams['figure.figsize'] = (12, 6)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Define paths - UPDATE WITH YOUR PATH\n",
            "DATA_DIR = Path(r'C:\\Users\\VIJAY BHUSHAN SINGH\\depression_detection_project\\data\\raw\\DAIC-WOZ')\n",
            "TRAIN_CSV = DATA_DIR / 'train_split_Depression_AVEC2017.csv'\n",
            "\n",
            "print(f'Data directory: {DATA_DIR}')\n",
            "print(f'Exists: {DATA_DIR.exists()}')"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load training data\n",
            "if TRAIN_CSV.exists():\n",
            "    train_df = pd.read_csv(TRAIN_CSV)\n",
            "    print('Train data loaded!')\n",
            "    print(f'Shape: {train_df.shape}')\n",
            "    display(train_df.head())\n",
            "else:\n",
            "    print('❌ File not found. Download dataset first.')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Next: Download dataset and extract features"]
    }
]

# Notebook 3: Audio Features
audio_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["# Audio Feature Extraction"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import librosa\n",
            "import librosa.display\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "from pathlib import Path"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load sample audio\n",
            "AUDIO_FILE = Path(r'C:\\Users\\VIJAY BHUSHAN SINGH\\depression_detection_project\\data\\raw\\DAIC-WOZ\\300_P\\300_AUDIO.wav')\n",
            "\n",
            "if AUDIO_FILE.exists():\n",
            "    audio, sr = librosa.load(AUDIO_FILE, sr=None)\n",
            "    print(f'Loaded: {AUDIO_FILE.name}')\n",
            "    print(f'Duration: {len(audio)/sr:.2f} seconds')\n",
            "else:\n",
            "    print('❌ Audio file not found')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Extract MFCCs, pitch, and other features here"]
    }
]

# Create notebooks
os.makedirs(r'notebooks\exploratory', exist_ok=True)

create_notebook(r'notebooks\exploratory\01_environment_test.ipynb', env_test_cells)
create_notebook(r'notebooks\exploratory\02_data_exploration.ipynb', data_exploration_cells)
create_notebook(r'notebooks\exploratory\03_audio_features.ipynb', audio_cells)

print("\n" + "="*50)
print("✓ All notebooks created!")
print("="*50)
print("\nStart Jupyter with: jupyter lab")
