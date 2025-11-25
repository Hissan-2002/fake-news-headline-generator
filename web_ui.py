import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import base64
from pathlib import Path

# Page Configuration
st.set_page_config(
    page_title="Bluffify - AI Headline Generator",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Professional Compact Theme for Bluffify
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Sora:wght@400;600;700&display=swap');
    
    /* === GLOBAL RESET === */
    .stApp {
        background: linear-gradient(135deg, #0a0614 0%, #1a0f2e 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Remove default padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 1rem !important;
        max-width: 900px !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* === LOGO & HEADER === */
    .logo-header {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .logo-header img {
        height: 60px;
        width: auto;
    }
    
    .logo-header h1 {
        font-family: 'Sora', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .subtitle {
        text-align: center;
        color: rgba(255, 255, 255, 0.6);
        font-size: 1rem;
        font-weight: 400;
        margin: 0.5rem 0 2rem 0;
        line-height: 1.5;
    }
    
    .divider {
        height: 2px;
        width: 80px;
        background: linear-gradient(90deg, transparent, #7540ce, transparent);
        margin: 1rem auto 2rem auto;
    }
    
    /* === MODEL SELECTION === */
    .stRadio {
        background: transparent !important;
    }
    
    .stRadio > label {
        font-family: 'Sora', sans-serif;
        font-size: 0.95rem;
        font-weight: 600;
        color: #7540ce;
        text-align: center;
        display: block;
        margin-bottom: 0.8rem;
    }
    
    .stRadio > div {
        display: flex;
        gap: 0.8rem;
        justify-content: center;
        flex-wrap: nowrap;
    }
    
    .stRadio > div > label {
        flex: 1;
        background: rgba(117, 64, 206, 0.08) !important;
        backdrop-filter: blur(10px) !important;
        border: 1.5px solid rgba(117, 64, 206, 0.25) !important;
        border-radius: 10px !important;
        padding: 0.9rem 1.2rem !important;
        cursor: pointer !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        color: rgba(255, 255, 255, 0.85) !important;
        text-align: center !important;
        min-width: 0;
        max-width: 200px;
    }
    
    .stRadio > div > label:hover {
        background: rgba(117, 64, 206, 0.15) !important;
        border-color: #7540ce !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(117, 64, 206, 0.3) !important;
    }
    
    .stRadio > div > label[data-baseweb="radio"]:has(input:checked) {
        background: linear-gradient(135deg, #7540ce 0%, #9d67e8 100%) !important;
        border-color: #7540ce !important;
        color: #ffffff !important;
        box-shadow: 0 4px 16px rgba(117, 64, 206, 0.5) !important;
    }
    
    /* === INPUT SECTION === */
    .input-section {
        background: rgba(20, 15, 35, 0.5);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(117, 64, 206, 0.2);
        border-radius: 12px;
        padding: 1.8rem;
        margin: 1.5rem 0;
    }
    
    .input-label {
        font-family: 'Sora', sans-serif;
        font-size: 1rem;
        font-weight: 600;
        color: rgba(255, 255, 255, 0.9);
        text-align: center;
        margin-bottom: 0.8rem;
    }
    
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1.5px solid rgba(117, 64, 206, 0.3) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
        font-size: 1rem !important;
        padding: 0.9rem 1.2rem !important;
        transition: all 0.25s ease !important;
        font-family: 'Inter', sans-serif !important;
        text-align: center !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #7540ce !important;
        box-shadow: 0 0 0 3px rgba(117, 64, 206, 0.15) !important;
        background: rgba(255, 255, 255, 0.08) !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: rgba(255, 255, 255, 0.4) !important;
    }
    
    .hint-text {
        text-align: center;
        color: rgba(255, 255, 255, 0.45);
        font-size: 0.85rem;
        margin: 0.6rem 0 1rem 0;
    }
    
    /* === BUTTON === */
    .stButton {
        display: flex;
        justify-content: center;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #7540ce 0%, #9d67e8 100%) !important;
        color: #ffffff !important;
        font-family: 'Sora', sans-serif !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.9rem 3rem !important;
        cursor: pointer !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 6px 20px rgba(117, 64, 206, 0.4) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        min-width: 280px !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 28px rgba(117, 64, 206, 0.6) !important;
        background: linear-gradient(135deg, #8654d9 0%, #a876f3 100%) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* === RESULT CARDS === */
    .result-card {
        background: rgba(20, 15, 35, 0.6);
        backdrop-filter: blur(20px);
        border: 1.5px solid rgba(117, 64, 206, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
        animation: fadeInUp 0.5s ease;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .result-card:hover {
        border-color: #7540ce;
        box-shadow: 0 8px 24px rgba(117, 64, 206, 0.3);
    }
    
    .model-chip {
        display: inline-block;
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        padding: 0.4rem 0.9rem;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
    
    .chip-finetuned {
        background: linear-gradient(135deg, #7540ce, #9d67e8);
        color: #ffffff;
    }
    
    .chip-base {
        background: rgba(120, 120, 140, 0.3);
        color: rgba(255, 255, 255, 0.85);
        border: 1px solid rgba(120, 120, 140, 0.4);
    }
    
    .headline-text {
        font-family: 'Inter', sans-serif;
        font-size: 1.25rem;
        line-height: 1.7;
        color: #ffffff;
        font-weight: 500;
        margin: 0.8rem 0;
    }
    
    .model-meta {
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        font-size: 0.8rem;
        color: rgba(255, 255, 255, 0.5);
        text-align: center;
    }
    
    /* === MESSAGES === */
    .stInfo, .stSuccess, .stError {
        background: rgba(20, 15, 35, 0.6) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 8px !important;
        border-left: 3px solid #7540ce !important;
        padding: 0.8rem 1rem !important;
        margin: 1rem 0 !important;
    }
    
    /* === EXPANDER === */
    .streamlit-expanderHeader {
        background: rgba(20, 15, 35, 0.5) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(117, 64, 206, 0.2) !important;
        color: rgba(255, 255, 255, 0.85) !important;
        font-family: 'Sora', sans-serif !important;
        font-size: 0.95rem !important;
        padding: 0.8rem 1rem !important;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #7540ce !important;
    }
    
    /* === LOADER === */
    .loader-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 1.5rem 0;
    }
    
    .loader {
        display: flex;
        gap: 0.5rem;
    }
    
    .loader span {
        width: 10px;
        height: 10px;
        background: #7540ce;
        border-radius: 50%;
        animation: bounce 1.2s infinite ease-in-out;
    }
    
    .loader span:nth-child(1) { animation-delay: -0.24s; }
    .loader span:nth-child(2) { animation-delay: -0.12s; }
    
    @keyframes bounce {
        0%, 80%, 100% { transform: scale(0); opacity: 0.5; }
        40% { transform: scale(1); opacity: 1; }
    }
    
    .loader-text {
        margin-top: 0.8rem;
        color: rgba(255, 255, 255, 0.5);
        font-size: 0.9rem;
    }
    
    /* === FOOTER === */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        margin-top: 2.5rem;
        border-top: 1px solid rgba(117, 64, 206, 0.15);
    }
    
    .footer p {
        color: rgba(255, 255, 255, 0.45);
        font-size: 0.85rem;
        margin: 0.3rem 0;
    }
    
    .footer a {
        color: #7540ce;
        text-decoration: none;
        font-weight: 500;
        transition: color 0.2s;
    }
    
    .footer a:hover {
        color: #9d67e8;
    }
    
    /* === RESPONSIVE === */
    @media (max-width: 768px) {
        .logo-header h1 { font-size: 2rem; }
        .logo-header img { height: 45px; }
        .stRadio > div { flex-direction: column; }
        .stRadio > div > label { max-width: 100%; }
    }
</style>
""", unsafe_allow_html=True)

# Cache models separately
@st.cache_resource
def load_finetuned_model():
    """Load fine-tuned model if available"""
    fine_tuned_paths = [
        ("./fine_tuned_model/final", "Fine-Tuned Model"),
        ("./fine_tuned_model_medium/final", "Fine-Tuned Model (Medium)"),
        ("./fine_tuned_model_large/final", "Fine-Tuned Model (Large)"),
    ]
    
    for path, name in fine_tuned_paths:
        if os.path.exists(path):
            tokenizer = AutoTokenizer.from_pretrained(path)
            model = AutoModelForCausalLM.from_pretrained(path)
            tokenizer.pad_token = tokenizer.eos_token
            
            model_info = {
                "name": name,
                "is_finetuned": True
            }
            return model, tokenizer, model_info
    
    return None, None, None

@st.cache_resource
def load_base_model():
    """Load GPT-2 Large base model"""
    tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    model = AutoModelForCausalLM.from_pretrained("gpt2-large")
    tokenizer.pad_token = tokenizer.eos_token
    
    model_info = {
        "name": "Base Model",
        "is_finetuned": False
    }
    
    return model, tokenizer, model_info

def generate_headline(topic, model, tokenizer, model_info):
    """Generate fake news headline"""
    is_finetuned = model_info["is_finetuned"]
    
    if is_finetuned:
        prompt = f"Fake News: {topic.title()}"
    else:
        prompt = f"Write a funny, satirical, absurd fake news headline about {topic}. Make it outrageous and super funny. Headline:"
    
    # Tokenize with attention_mask to avoid warnings
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=inputs["input_ids"].shape[1] + (60 if is_finetuned else 40),
            do_sample=True,
            temperature=1.1 if is_finetuned else 1.5,
            top_p=0.95,
            top_k=100,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id
        )
    
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    headline = full_output.replace(prompt, "").strip()
    
    if "Headline:" in headline:
        headline = headline.split("Headline:")[-1].strip()
    
    headline = headline.split('\n')[0].strip().lstrip(':- "\'')
    
    if headline and not headline.endswith(('.', '!', '?')):
        headline += "!"
    
    return headline if headline else "Breaking: Something Absolutely Ridiculous Just Happened!"

# Main UI
def main():
    # Load models
    with st.spinner("Loading AI models..."):
        finetuned_model, finetuned_tokenizer, finetuned_info = load_finetuned_model()
        base_model, base_tokenizer, base_info = load_base_model()
    
    # Header with Logo
    logo_path = "BluffifyLogo.png"
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            logo_data = base64.b64encode(f.read()).decode()
        st.markdown(f"""
        <div class="logo-header">
            <img src="data:image/png;base64,{logo_data}" alt="Bluffify Logo">
            <h1>Bluffify</h1>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="logo-header"><h1>Bluffify</h1></div>', unsafe_allow_html=True)
    
    st.markdown('<p class="subtitle">Generate hilariously absurd headlines with AI-powered models</p>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Model Selection (Compact)
    if finetuned_info:
        mode = st.radio(
            "Choose Model",
            ["Fine-Tuned Model", "Base Model", "Compare Both"],
            horizontal=True,
            label_visibility="visible"
        )
    else:
        mode = "Base Model"
        st.info("Fine-tuned model not available. Using base model.")
    
    # Input Section (Centered & Compact)
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<div class="input-label">Enter Your Topic</div>', unsafe_allow_html=True)
    
    topic = st.text_input(
        "Topic",
        placeholder="e.g., robots, pizza, crypto, exams...",
        label_visibility="collapsed"
    )
    
    st.markdown('<p class="hint-text">Try: aliens, politicians, coffee, AI</p>', unsafe_allow_html=True)
    
    generate_btn = st.button("Generate Headline", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Results Section
    if generate_btn and topic.strip():
        # Loader placeholder
        loader_placeholder = st.empty()
        loader_placeholder.markdown("""
        <div class="loader-container">
            <div class="loader">
                <span></span><span></span><span></span>
            </div>
            <div class="loader-text">Generating...</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Compare Both Mode
        if mode == "Compare Both" and finetuned_info:
            # Clear loader before showing results
            loader_placeholder.empty()
            col1, col2 = st.columns(2, gap="medium")
            
            with col1:
                headline_ft = generate_headline(topic, finetuned_model, finetuned_tokenizer, finetuned_info)
                st.markdown(f"""
                <div class="result-card">
                    <span class="model-chip chip-finetuned">Fine-Tuned</span>
                    <div class="headline-text">{headline_ft}</div>
                    <div class="model-meta">8,000 headlines | 124M params</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                headline_base = generate_headline(topic, base_model, base_tokenizer, base_info)
                st.markdown(f"""
                <div class="result-card">
                    <span class="model-chip chip-base">Base Model</span>
                    <div class="headline-text">{headline_base}</div>
                    <div class="model-meta">GPT-2 Large | 774M params</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Single Model Mode
        else:
            if mode == "Fine-Tuned Model" and finetuned_info:
                model, tokenizer, model_info = finetuned_model, finetuned_tokenizer, finetuned_info
                chip_class = "chip-finetuned"
                chip_label = "Fine-Tuned"
                meta_text = "8,000 headlines | 124M params"
            else:
                model, tokenizer, model_info = base_model, base_tokenizer, base_info
                chip_class = "chip-base"
                chip_label = "Base Model"
                meta_text = "GPT-2 Large | 774M params"
            
            headline = generate_headline(topic, model, tokenizer, model_info)
            
            # Clear loader before showing results
            loader_placeholder.empty()
            
            st.markdown(f"""
            <div class="result-card">
                <span class="model-chip {chip_class}">{chip_label}</span>
                <div class="headline-text">{headline}</div>
                <div class="model-meta">{meta_text}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.info("✨ All headlines are AI-generated fiction for entertainment only.")
    
    elif generate_btn:
        st.error("Please enter a topic first.")
    
    # Compact Info Section
    with st.expander("ℹ️ About Bluffify"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Fine-Tuned Model**
            - Architecture: GPT-2 Base
            - Parameters: 124M
            - Training: 8,000 headlines
            - Style: Authentic fake news
            """)
        
        with col2:
            st.markdown("""
            **Base Model**
            - Architecture: GPT-2 Large
            - Parameters: 774M
            - Training: General web text
            - Style: Creative generation
            """)
        
        st.markdown("""
        ---
        **How It Works:** Bluffify uses transformer-based AI models to generate satirical headlines. 
        The Fine-Tuned model specializes in authentic fake news style, while the Base model offers 
        creative variety. Compare Both mode shows results side-by-side.
        """)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Built with Streamlit & HuggingFace Transformers</p>
        <p><a href="https://github.com/Hissan-2002/fake-news-headline-generator" target="_blank">View on GitHub</a></p>
        <p>© 2025 Bluffify - Entertainment purposes only</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
