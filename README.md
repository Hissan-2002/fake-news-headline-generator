# 📰 Fake News Headline Generator

A fun AI-powered web application that generates humorous and exaggerated fake news headlines based on user-provided topics using GPT-2 from HuggingFace.

## 🎯 Project Description

This project uses the GPT-2 Large language model (774M parameters) to generate creative and entertaining fake news headlines. Simply enter any topic, and the AI will create a headline that's both amusing and completely fictional. The project features a clean, user-friendly web interface built with Streamlit.

**Note:** This is a minimal implementation that works out-of-the-box. No dataset or fine-tuning is required to run the project, though you can optionally fine-tune the model on your own dataset if desired.

## 🚀 Features

- Clean, centered web UI with Streamlit
- Real-time headline generation using GPT-2
- Fast inference with model caching
- No training or dataset required
- Entertainment-focused output with creative parameters

## 📋 Requirements

- Python 3.7 or higher
- Internet connection (for first-time model download)

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
   - `torch` - PyTorch for model inference
   - `streamlit` - Web UI framework

## ▶️ Running the Application

To start the web UI, run:

```bash
streamlit run web_ui.py
```

The application will:
- Load the GPT-2 model (first run may take a minute to download)
- Open in your default web browser at `http://localhost:8501`
- Be ready to generate headlines!

## 💡 How to Use

1. Enter any topic in the text input field (e.g., "cats", "technology", "politics")
2. Click the "Generate Headline" button
3. Watch as the AI creates a humorous fake news headline
4. Generate as many headlines as you want!

## 🎨 Generation Parameters

The model uses these parameters for creative output:
- `max_length=30` - Keeps headlines concise
- `do_sample=True` - Enables creative sampling
- `temperature=0.9` - High creativity/randomness
- `top_p=0.92` - Nucleus sampling for quality

## 📁 Project Structure

```
fake-news-generator/
│── web_ui.py          # Streamlit web UI (main application)
│── requirements.txt   # Python dependencies
│── README.md         # This file
```

## ⚙️ Optional: Fine-tuning

While the project works perfectly without any training, you can optionally fine-tune the model on your own dataset of news headlines for more specific outputs. This is completely optional and not required for basic functionality.

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
