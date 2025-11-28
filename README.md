# 🎭 Bluffify - AI Fake News Generator

AI-powered web application that generates hilariously satirical fake news headlines using state-of-the-art fine-tuned GPT-2 models with LoRA optimization.

## 🌟 Project Overview

Bluffify uses transformer-based language models to generate humorous and absurd fake news headlines. The application features **three different models** including a cutting-edge LoRA fine-tuned model trained on 8,000 authentic fake news headlines, providing the perfect balance of creativity and humor.

## ✨ Features

- **🎯 Triple Model System**: Choose between Old Fine-Tuned, LoRA Fine-Tuned (New), or Base GPT-2
- **🔄 Compare All Models**: See all three models generate headlines side-by-side
- **⚡ LoRA Fine-Tuned Model**: Parameter-efficient training preserves creativity while learning fake news patterns
- **🎨 Beautiful UI**: Modern dark theme with purple branding (#7540ce)
- **🎭 Bluffify Branding**: Custom logo and professional design
- **📊 Model Status Indicator**: Real-time display of active models
- **🎪 Humor-Optimized**: Multi-attempt generation with comedy scoring

## 🤖 Model Details

### LoRA Fine-Tuned Model ⭐ (Recommended)
- **Architecture**: GPT-2 Base (124M parameters) + LoRA
- **Trainable Parameters**: 1.6M (1.3% of total)
- **Training Data**: 8,000 fake news headlines
- **Training Duration**: ~2.5 hours (GPU: NVIDIA GTX 1650 Ti)
- **Epochs**: 5
- **Final Loss**: 3.0189
- **Training Date**: November 28, 2025
- **Method**: Parameter-Efficient Fine-Tuning (PEFT) with LoRA
- **Location**: `./fine_tuned_model_lora/final_merged/`
- **Benefits**: Best topic relevance, preserves creativity, most humorous

### Old Fine-Tuned Model
- **Architecture**: GPT-2 Base (124M parameters)
- **Trainable Parameters**: 124M (100% of total)
- **Training Data**: 8,000 fake news headlines
- **Epochs**: 3
- **Final Loss**: 3.0238
- **Method**: Full fine-tuning
- **Location**: `./fine_tuned_model/final/`
- **Issues**: Overfitted on political news, poor context understanding

### Base GPT-2 Model
- **Architecture**: GPT-2 Base (124M parameters)
- **Source**: OpenAI via HuggingFace
- **Training**: Pre-trained on WebText corpus
- **Use Case**: Creative baseline comparison
- **Auto-downloads**: First run

## 📊 Dataset

- **Source**: Fake.csv (Kaggle Fake News Dataset)
- **Size**: 59.88 MB
- **Total Headlines**: 20,000+
- **Used for Training**: 8,000 headlines
- **Training Split**: 90% train (7,200), 10% validation (800)

## 🚀 Installation

### Prerequisites
- Python 3.13+ (tested with 3.13.9)
- 8GB+ RAM
- 5GB free disk space
- GPU recommended for training (NVIDIA GTX 1650 Ti or better)

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/Hissan-2002/fake-news-headline-generator.git
cd fake-news-headline-generator
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Dependencies
- `transformers>=4.30.0` - HuggingFace Transformers
- `torch>=2.7.1+cu118` - PyTorch with CUDA 11.8 support
- `streamlit>=1.28.0` - Web UI framework
- `datasets>=2.14.0` - Dataset library
- `accelerate>=0.21.0` - Training acceleration
- `pandas>=2.0.0` - Data processing
- `scikit-learn>=1.3.0` - Data utilities
- `peft>=0.7.0` - Parameter-Efficient Fine-Tuning
- `bitsandbytes>=0.41.0` - Quantization support

## 💻 Usage

### Running the Application

```bash
streamlit run web_ui.py
```

The application will open at `http://localhost:8501`

### Using the Interface

1. **Check Model Status**: View which models are active (green ✓)
2. **Choose Model**: Select from 4 options:
   - **LoRA Fine-Tuned (New)** ⭐ - Best results
   - **Old Fine-Tuned** - Legacy model
   - **Base GPT-2** - Vanilla creativity
   - **Compare All Models** - See all outputs side-by-side
3. **Enter Topic**: Type any word (pizza, robots, cryptocurrency, etc.)
4. **Generate**: Click "Generate Headline"
5. **Enjoy**: Read the hilarious AI-generated headline!

### Model Comparison Mode

Select **"Compare All Models"** to see outputs from all available models in equal-width columns with beautiful staggered animations.

## 🎓 Training the LoRA Model

### Why LoRA?
**LoRA (Low-Rank Adaptation)** is a parameter-efficient fine-tuning method that:
- Trains only **1.3%** of parameters (1.6M vs 124M)
- Preserves base model's language understanding
- Prevents catastrophic forgetting
- Better generalization to new topics
- 3x faster training than full fine-tuning

### Training Script

```bash
python train_lora_optimized.py
```

### Training Configuration

```python
# Model & Data
MODEL_NAME = "gpt2"
SAMPLE_SIZE = 8000
EPOCHS = 5
BATCH_SIZE = 8
MAX_LENGTH = 128
LEARNING_RATE = 2e-4

# LoRA Configuration
LoraConfig(
    r=16,                           # Rank
    lora_alpha=32,                  # Scaling factor
    target_modules=["c_attn", "c_proj", "c_fc"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
```

### Training Process

1. **Data Loading**: Loads and filters Fake.csv
2. **Keyword Extraction**: Extracts topics from headlines (35+ categories)
3. **Prompt Engineering**: Creates 5 diverse prompt variations
4. **LoRA Application**: Applies adapters to attention layers
5. **Training**: Fine-tunes for 5 epochs with GPU acceleration
6. **Merging**: Merges LoRA weights with base model
7. **Saving**: Saves to `./fine_tuned_model_lora/final_merged/`

### Training Time

- **GPU (GTX 1650 Ti)**: ~2.5 hours (140 minutes)
- **CPU**: ~55-70 minutes
- **Training Steps**: 4,500 steps (5 epochs × 900 steps/epoch)

### Hardware Used

- **GPU**: NVIDIA GeForce GTX 1650 Ti (4GB VRAM)
- **CUDA**: 11.8
- **PyTorch**: 2.7.1+cu118
- **Training Speed**: ~1.88s per iteration

## 🎪 Generation Parameters

### Humor-Optimized Settings
```python
# Multi-attempt with comedy scoring
attempts = 3
temperature = 0.9 - 1.0  # High for creativity
top_p = 0.92
top_k = 50
repetition_penalty = 1.8
max_length = 35 tokens

# Humor scoring factors
- Topic relevance (+20 points)
- Funny words (+5 each): "declares", "reveals", "shocked", etc.
- Good length 30-120 chars (+15)
- Has numbers (+3)
- No garbage text (+10)
```

### Funny Word Detection
Awards bonus points for: declares, reveals, scientists, discovers, study, experts, shocked, horrified, demands, accidentally, secretly, refuses, admits, claims, warns, predicts

### Fallback Templates
If generation fails, uses hilarious templates:
- "Scientists Discover {topic} Can Cure Monday Blues"
- "Experts Warn: Too Much {topic} May Cause Spontaneous Dancing"
- "Study: 10 Out of 10 Cats Prefer {topic} Over World Peace"

## 📁 Project Structure

```
fake-news-generator/
├── web_ui.py                    # Main Streamlit application ⭐
├── train_lora_optimized.py      # LoRA training script
├── BluffifyLogo.png             # App logo
├── Fake.csv                     # Training dataset (59.88 MB)
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── .gitignore                   # Git ignore rules
├── .gitattributes               # Git LFS configuration
├── fine_tuned_model/            # Old fine-tuned model
│   └── final/
│       ├── config.json
│       ├── model.safetensors
│       └── tokenizer files
└── fine_tuned_model_lora/       # LoRA fine-tuned model
    └── final_merged/            # Merged model (used by app)
        ├── config.json
        ├── model.safetensors
        └── tokenizer files
```

## 🎨 Technical Architecture

### UI Design
- **Theme**: Dark mode with purple accent (#7540ce)
- **Fonts**: Space Grotesk (headings), Poppins (body)
- **Logo**: Custom Bluffify branding
- **Status Indicators**: Pulsing animations for active models
- **Cards**: Glassmorphism with backdrop blur

### Generation Pipeline

1. **Input**: User enters topic
2. **Prompt Engineering**: Creates humor-focused prompts
3. **Multi-Attempt**: Generates 3 variations with different temperatures
4. **Aggressive Cleaning**: Removes [VIDEO], URLs, garbage text
5. **Comedy Scoring**: Rates each attempt on relevance and humor
6. **Best Selection**: Returns highest-scoring headline
7. **Fallback**: Uses template if all attempts fail

### LoRA Architecture

```
GPT-2 Base (124M params)
    ├── Frozen layers (98.7%)
    └── LoRA adapters (1.3%)
        ├── c_attn (attention)
        ├── c_proj (projection)
        └── c_fc (feed-forward)
```

## 📊 Performance

### Generation Speed
- **LoRA Fine-Tuned**: ~3-4 seconds
- **Old Fine-Tuned**: ~3-4 seconds
- **Base GPT-2**: ~2-3 seconds
- **Compare All**: ~10-12 seconds (sequential)

### Quality Comparison
| Model | Topic Relevance | Humor | Creativity | Style |
|-------|----------------|-------|------------|-------|
| **LoRA Fine-Tuned** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Authentic |
| **Old Fine-Tuned** | ⭐⭐ | ⭐⭐ | ⭐⭐ | Overfitted |
| **Base GPT-2** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Generic |

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce `BATCH_SIZE` to 4 or 2
- Reduce `MAX_LENGTH` to 96
- Close other GPU applications

### Model Not Loading
- Check if `fine_tuned_model_lora/final_merged/` exists
- Verify PyTorch CUDA installation: `torch.cuda.is_available()`
- Restart Streamlit app

### Poor Headline Quality
- Try "LoRA Fine-Tuned (New)" model
- Use simpler single-word topics
- Regenerate multiple times

### Slow Generation on CPU
- First run downloads ~3GB base model
- Subsequent runs use cached model
- Consider using GPU for 3-4x speedup

## 🔧 Advanced Configuration

### Custom Training Topics
Edit keyword extraction in `train_lora_optimized.py`:
```python
topics = [
    # Add your custom topics here
    'YourTopic1', 'YourTopic2', ...
]
```

### Adjust Humor Settings
Modify generation parameters in `web_ui.py`:
```python
temperature = 1.0  # Lower = more coherent, Higher = more creative
funny_words = ['add', 'your', 'funny', 'words']
```

## ⚠️ Disclaimer

**This application generates fictional satirical content for entertainment purposes only.** All headlines are AI-created fiction and should not be taken as factual information. Use responsibly and ethically.

## 🙏 Acknowledgments

- **Dataset**: Kaggle Fake News Dataset
- **Framework**: HuggingFace Transformers & PEFT
- **Base Model**: OpenAI GPT-2
- **UI Framework**: Streamlit
- **Training Method**: LoRA (Low-Rank Adaptation)
- **GPU Support**: PyTorch with CUDA

## 📜 License

This project is for educational and entertainment purposes only.

## 🔗 Repository

- **GitHub**: https://github.com/Hissan-2002/fake-news-headline-generator
- **Owner**: Hissan-2002
- **Branch**: main

## 📝 Version History

- **v2.0** (Nov 28, 2025): LoRA model, 3-model system, Bluffify branding
  - Added LoRA fine-tuned model with 1.3% parameter training
  - Implemented 3-model comparison (Old, LoRA, Base)
  - Rebranded to Bluffify with custom logo
  - Added humor-optimized multi-attempt generation
  - Implemented comedy scoring system
  - GPU acceleration with CUDA 11.8
  - Aggressive output cleaning and validation
  
- **v1.0** (Nov 25, 2025): Initial release
  - Fine-tuned GPT-2 on 8,000 headlines
  - Dual model comparison (Fine-tuned vs Base)
  - Clean Streamlit UI

## 🎯 What Makes Bluffify Special?

1. **LoRA Technology**: Only trains 1.3% of parameters while preserving creativity
2. **Triple Model System**: Compare different approaches side-by-side
3. **Humor Optimization**: Multi-attempt generation with comedy scoring
4. **Context Awareness**: Understands topics across 35+ categories
5. **Professional UI**: Beautiful dark theme with custom branding
6. **Smart Fallbacks**: Hilarious templates when generation fails
7. **Aggressive Cleaning**: Removes garbage text for pristine output

---

**Generate hilariously absurd headlines with Bluffify! 🎭✨**

Made with 💜 using LoRA, GPT-2, and lots of humor.
