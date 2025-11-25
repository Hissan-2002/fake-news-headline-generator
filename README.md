# 📰 Fake News Headline Generator

A fun AI-powered web application that generates humorous and exaggerated fake news headlines based on user-provided topics using **Fine-Tuned GPT-2 Large** trained on a real fake news dataset from Kaggle.

## 🎯 Project Description

This project uses GPT-2 Large (774M parameters) fine-tuned on a comprehensive fake news dataset to generate creative and entertaining fake news headlines. The model has been trained on thousands of real fake news headlines to understand the style, tone, and absurdity that makes them compelling. Simply enter any topic, and the AI will create a headline that's both amusing and completely fictional. The project features a clean, user-friendly web interface built with Streamlit.

**Key Features:**
- 🤖 Fine-tuned GPT-2 Large model trained on fake news dataset
- 🎨 Clean, intuitive Streamlit web interface
- ⚡ Fast inference with model caching
- 🎭 Generates authentic-style fake news headlines
- 📊 Trained on real data for maximum authenticity

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
