# 🚀 Quick Start Guide

## First Time Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model (Required for best results)
```bash
python train_model.py
```
⏱️ **Time:** 2-4 hours with GPU, 8-12 hours with CPU

**What happens:**
- Loads `Fake.csv` dataset
- Extracts headlines from 'title' column
- Fine-tunes GPT-2 Large for 3 epochs
- Saves model to `./fine_tuned_model/final`

### 3. Test the Model (Optional)
```bash
python test_model.py
```
Generates sample headlines to verify everything works.

### 4. Run the Web UI
```bash
python -m streamlit run web_ui.py
```
Opens at `http://localhost:8501`

---

## Quick Training Test (5-10 minutes)

Want to test the pipeline quickly? Edit `train_model.py`:

```python
# Line ~60 - Add sample_size parameter
dataset = load_and_prepare_data(DATASET_FILE, sample_size=500)

# Line ~11 - Reduce epochs
EPOCHS = 1
```

Then run:
```bash
python train_model.py
```

This creates a basic fine-tuned model quickly for testing.

---

## Without Training

The app works without training but uses the base GPT-2 Large model:

```bash
python -m streamlit run web_ui.py
```

You'll see: ⚠️ "Fine-tuned model not found. Using base GPT-2 Large."

Results will be less authentic but still functional.

---

## Troubleshooting

**CUDA Out of Memory?**
```python
# In train_model.py
BATCH_SIZE = 2  # or even 1
```

**Training too slow?**
```python
# Use fewer samples
dataset = load_and_prepare_data(DATASET_FILE, sample_size=1000)
EPOCHS = 1
```

**Can't find streamlit command?**
```bash
python -m streamlit run web_ui.py
```

---

## File Sizes to Expect

- `Fake.csv`: ~50-150MB (your dataset)
- Base GPT-2 Large download: ~3GB (one-time)
- Fine-tuned model: ~3GB
- Total disk space needed: ~10GB

---

## What Gets Generated

**Example inputs → outputs:**

| Input | Sample Output |
|-------|--------------|
| cats | "Scientists Baffled as Cats Declare Independence, Form New Government" |
| pizza | "Breaking: Pizza Delivery Guy Discovers Time Travel While Making Deliveries" |
| aliens | "Government Finally Admits Aliens Have Been Running Social Media All Along" |
| politics | "Local Politician Caught Teaching Pigeons to Vote in Upcoming Election" |

---

## Next Steps After Setup

1. ✅ Train the model (`python train_model.py`)
2. ✅ Test it (`python test_model.py`)
3. ✅ Run the UI (`python -m streamlit run web_ui.py`)
4. 🎉 Generate hilarious headlines!
5. 📤 Share with friends

---

**Need Help?** Check the full README.md for detailed documentation.
