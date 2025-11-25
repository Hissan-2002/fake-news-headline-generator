# Fake News Headline Generator

AI-powered web application that generates satirical fake news headlines using fine-tuned GPT-2 models.

## Project Overview

This project uses transformer-based language models to generate humorous and exaggerated fake news headlines. The application includes a fine-tuned model trained on 8,000 real fake news headlines from Kaggle, allowing for authentic-style headline generation.

## Features

- **Dual Model System**: Compare fine-tuned vs base GPT-2 models
- **Side-by-Side Comparison**: See both models generate headlines simultaneously
- **Fine-Tuned Model**: Trained specifically on fake news dataset for authentic style
- **Base Model**: Standard GPT-2 Large for creative outputs
- **Clean Web Interface**: Built with Streamlit for easy interaction

## Model Details

### Fine-Tuned Model
- **Base Architecture**: GPT-2 Base (124M parameters)
- **Training Data**: 8,000 fake news headlines from Kaggle dataset
- **Training Duration**: ~2 hours 43 minutes (on CPU)
- **Epochs**: 3
- **Final Loss**: 3.0238
- **Training Date**: November 25, 2025
- **Location**: `./fine_tuned_model/final/`

### Base Model
- **Architecture**: GPT-2 Large (774M parameters)
- **Source**: HuggingFace Transformers
- **Training**: Pre-trained on general internet text
- **Use Case**: Comparison baseline

## Dataset

- **Source**: Fake.csv (Kaggle Fake News Dataset)
- **Size**: 59.88 MB
- **Total Headlines**: 20,000+
- **Used for Training**: 8,000 headlines
- **Columns**: title, text, subject, date
- **Training Split**: 90% train, 10% validation

## Installation

### Prerequisites
- Python 3.7+
- 8GB+ RAM
- 10GB free disk space

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Hissan-2002/fake-news-headline-generator.git
cd fake-news-headline-generator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies
- `transformers>=4.30.0` - HuggingFace Transformers library
- `torch>=2.0.0` - PyTorch for model training and inference
- `streamlit>=1.28.0` - Web UI framework
- `datasets>=2.14.0` - HuggingFace datasets library
- `accelerate>=0.21.0` - Training acceleration
- `pandas>=2.0.0` - Data processing
- `scikit-learn>=1.3.0` - Data utilities

## Usage

### Running the Application

```bash
streamlit run web_ui.py
```

Or:
```bash
python -m streamlit run web_ui.py
```

The application will open at `http://localhost:8501`

### Using the Interface

1. **Select Model**: Choose Fine-Tuned Model, Base Model, or Compare Both
2. **Enter Topic**: Type any word or phrase (e.g., "robots", "politics", "cats")
3. **Generate**: Click the Generate button
4. **View Results**: See the generated headline(s)

### Model Comparison

Select "Compare Both" mode to see outputs from both models side-by-side, demonstrating the difference between fine-tuned and base models.

## Training

### Training the Fine-Tuned Model

```bash
python train_optimized.py
```

### Training Configuration

```python
MODEL_NAME = "gpt2"  # GPT-2 Base
SAMPLE_SIZE = 8000   # Training samples
EPOCHS = 3           # Training epochs
BATCH_SIZE = 4       # Batch size
MAX_LENGTH = 80      # Max token length
LEARNING_RATE = 5e-5 # Learning rate
```

### Training Process

1. **Data Loading**: Loads Fake.csv dataset
2. **Preprocessing**: Filters headlines, removes duplicates
3. **Tokenization**: Converts text to model inputs
4. **Training**: Fine-tunes for 3 epochs
5. **Validation**: Evaluates on 10% validation set
6. **Saving**: Saves model to `./fine_tuned_model/final/`

### Training Metrics

- **Training Loss**: 3.0238
- **Evaluation Loss**: 2.9665
- **Training Samples**: 7,200
- **Validation Samples**: 800
- **Total Steps**: 2,700
- **Checkpoints**: Saved every 1,000 steps

### Hardware Requirements

**Minimum (CPU Training):**
- CPU: Multi-core processor
- RAM: 8GB
- Time: 60-90 minutes

**Recommended (GPU Training):**
- GPU: 6GB+ VRAM
- RAM: 16GB
- Time: 15-20 minutes

## Generation Parameters

### Fine-Tuned Model
```python
temperature = 1.1        # Creativity level
top_p = 0.95            # Nucleus sampling
top_k = 100             # Top-k sampling
max_tokens = 60         # Maximum output length
repetition_penalty = 1.2 # Reduce repetition
```

### Base Model
```python
temperature = 1.5        # Higher creativity
top_p = 0.95
top_k = 100
max_tokens = 40
repetition_penalty = 1.2
```

## Project Structure

```
fake-news-generator/
├── web_ui.py              # Main Streamlit application
├── train_optimized.py     # Model training script
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── .gitignore            # Git ignore rules
├── Fake.csv              # Training dataset (59.88 MB)
└── fine_tuned_model/     # Fine-tuned model directory
    └── final/
        ├── config.json
        ├── pytorch_model.bin
        ├── tokenizer.json
        ├── vocab.json
        └── merges.txt
```

## Technical Architecture

### Model Pipeline

1. **Input Processing**: User enters topic → Text preprocessing
2. **Prompt Engineering**: Format differs for fine-tuned vs base model
3. **Generation**: Model generates tokens autoregressively
4. **Post-Processing**: Clean output, extract headline
5. **Display**: Show formatted result to user

### Fine-Tuning Process

1. **Data Preparation**: Load and clean fake news headlines
2. **Tokenization**: Convert text to token IDs
3. **Training Loop**: Update model weights over 3 epochs
4. **Validation**: Evaluate on held-out data
5. **Model Saving**: Export trained model and tokenizer

## Performance

### Generation Speed
- Fine-Tuned Model: ~2-3 seconds per headline
- Base Model: ~3-4 seconds per headline
- Comparison Mode: ~5-7 seconds (sequential generation)

### Quality Metrics
- Fine-Tuned Model: Authentic fake news style, higher coherence
- Base Model: More creative but less authentic style

## API Reference

### Main Functions

#### `load_finetuned_model()`
Loads the fine-tuned model from disk.
- **Returns**: model, tokenizer, model_info dict
- **Caches**: Uses Streamlit cache for performance

#### `load_base_model()`
Loads GPT-2 Large from HuggingFace.
- **Returns**: model, tokenizer, model_info dict

#### `generate_headline(topic, model, tokenizer, model_info)`
Generates a headline for the given topic.
- **Parameters**: 
  - topic: Input topic string
  - model: Loaded model
  - tokenizer: Model tokenizer
  - model_info: Model metadata dict
- **Returns**: Generated headline string

## Troubleshooting

### Common Issues

**Model not found:**
- Ensure fine-tuned model exists in `./fine_tuned_model/final/`
- Run `train_optimized.py` to train the model

**Out of memory:**
- Reduce batch size in training script
- Use CPU instead of GPU
- Close other applications

**Slow generation:**
- First run downloads base model (~3GB)
- Subsequent runs use cached model
- GPU significantly faster than CPU

## License

This project is for educational and entertainment purposes only.

## Acknowledgments

- **Dataset**: Kaggle Fake News Dataset
- **Framework**: HuggingFace Transformers
- **Base Model**: OpenAI GPT-2
- **UI Framework**: Streamlit

## Disclaimer

This application generates fictional content for entertainment purposes only. All generated headlines are AI-created and should not be taken as factual information.

## Repository

**GitHub**: https://github.com/Hissan-2002/fake-news-headline-generator
**Owner**: Hissan-2002
**Branch**: main

## Version History

- **v1.0** (Nov 25, 2025): Initial release with fine-tuned model
- Model trained on 8,000 headlines
- Dual model comparison feature
- Clean minimal UI

## 🚀 Features

- Fine-tuned GPT-2 Large trained on authentic fake news dataset
- Clean, centered web UI with Streamlit
- Real-time headline generation
- Intelligent fallback to base model if fine-tuned model unavailable
- Model caching for fast subsequent generations
- Entertainment-focused with creative parameter tuning

## 📋 Requirements

- Python 3.7 or higher
- Internet connection (for first-time model download)
- GPU recommended for training (CPU works but slower)
- ~10GB disk space for models and dataset

## 🛠️ Installation

1. **Clone or download this project**

2. **Navigate to the project directory:**
   ```bash
   cd fake-news-generator
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   This will install:
   - `transformers` - HuggingFace library for GPT-2
   - `torch` - PyTorch for model training and inference
   - `streamlit` - Web UI framework
   - `datasets` - HuggingFace datasets library
   - `accelerate` - Training acceleration
   - `pandas` - Data processing
   - `scikit-learn` - Data splitting utilities
   - `tqdm` - Progress bars

## 🎓 Training the Model (Required for Best Results)

Before running the web UI, you should fine-tune the model on the fake news dataset for authentic results:

**Note:** The `Fake.csv` file should be in the project directory. This contains the fake news dataset from Kaggle with columns: title, text, subject, date.

### Step 1: Train the model

```bash
python train_model.py
```

**Training Details:**
- Uses the `title` column from the fake news dataset
- Fine-tunes GPT-2 Large for 3 epochs (configurable)
- Creates train/validation split (90/10)
- Saves checkpoints every 1000 steps
- Final model saved to `./fine_tuned_model/final`

**Expected Training Time:**
- With GPU: 2-4 hours
- With CPU: 8-12 hours

**Training Options:**
You can modify these parameters in `train_model.py`:
- `EPOCHS`: Number of training epochs (default: 3)
- `BATCH_SIZE`: Batch size (default: 4, reduce if out of memory)
- `LEARNING_RATE`: Learning rate (default: 2e-5)
- `MAX_LENGTH`: Maximum sequence length (default: 128)

### Step 2: Test the fine-tuned model (Optional)

```bash
python test_model.py
```

This will generate sample headlines for various topics to verify the model works correctly.

## ▶️ Running the Application

To start the web UI, run:

```bash
streamlit run web_ui.py
```

**OR** (if streamlit is not in PATH):

```bash
python -m streamlit run web_ui.py
```

The application will:
- Check for fine-tuned model (shows warning if not found)
- Load the model (first run downloads ~3GB)
- Open in your default web browser at `http://localhost:8501`
- Be ready to generate headlines!

**Note:** The app works with or without fine-tuning, but **fine-tuned model produces much better results**.

## 💡 How to Use

1. Enter any topic in the text input field (e.g., "cats", "technology", "politics")
2. Click the "Generate Headline" button
3. Watch as the AI creates a humorous fake news headline
4. Generate as many headlines as you want!

## 🎨 Generation Parameters

The model uses these parameters for creative output:
- `temperature=1.3` (fine-tuned) / `1.5` (base) - High creativity/randomness
- `top_p=0.95` - Nucleus sampling for quality
- `top_k=100` - Top-k sampling for diversity
- `max_length=50` - Complete sentences
- `repetition_penalty=1.2` - Avoid repetition
- `no_repeat_ngram_size=3` - Prevent repetitive phrases

## 📁 Project Structure

```
fake-news-generator/
│── web_ui.py                # Streamlit web UI (main application)
│── train_model.py           # Model fine-tuning script
│── test_model.py            # Test fine-tuned model
│── requirements.txt         # Python dependencies
│── README.md                # This file
│── Fake.csv                 # Fake news dataset (from Kaggle)
│── .gitignore              # Git ignore rules
│
└── fine_tuned_model/        # Created after training
    └── final/               # Final fine-tuned model
        ├── config.json
        ├── pytorch_model.bin
        └── tokenizer files
```

## 🔧 Advanced Configuration

### Adjusting Training Parameters

Edit `train_model.py` to customize training:

```python
# Quick training (for testing)
EPOCHS = 1
BATCH_SIZE = 2

# Production training (better quality)
EPOCHS = 3-5
BATCH_SIZE = 4-8 (depending on GPU memory)
```

### Using a Subset of Data

For quick experimentation, modify the dataset loading:

```python
# In train_model.py, line ~60
dataset = load_and_prepare_data(DATASET_FILE, sample_size=1000)  # Use only 1000 samples
```

## 🐛 Troubleshooting

### CUDA Out of Memory
If you get GPU memory errors during training:
- Reduce `BATCH_SIZE` in `train_model.py` (try 2 or 1)
- Reduce `MAX_LENGTH` (try 64 instead of 128)
- Use gradient accumulation

### Training Too Slow
- Use a smaller subset of data initially
- Reduce number of epochs
- Consider using Google Colab with free GPU

### Model Not Found Warning
If you see "Fine-tuned model not found" in the UI:
- Run `python train_model.py` first
- Wait for training to complete
- Restart Streamlit app

## ⚠️ Disclaimer

This project generates fictional content for entertainment purposes only. The headlines are not real news and should not be treated as factual information.

## 🤝 Credits

- Built with [HuggingFace Transformers](https://huggingface.co/transformers/)
- Powered by [GPT-2 Large](https://huggingface.co/gpt2-large) (774M parameters)
- UI created with [Streamlit](https://streamlit.io/)

## 📝 License

This project is open-source and available for educational and entertainment purposes.

---

**Enjoy generating fake news headlines responsibly! 📰✨**
