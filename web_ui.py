import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import base64
from pathlib import Path

# Page Configuration
st.set_page_config(
    page_title="Bluffify - AI Fake News Generator",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional Dark Theme with #7540ce Brand Color
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');
    
    /* === GLOBAL THEME === */
    .stApp {
        background: linear-gradient(135deg, #0f0c1e 0%, #1a1635 100%);
        background-attachment: fixed;
        font-family: 'Poppins', sans-serif;
    }
    
    /* Animated Background Gradient */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(circle at 15% 20%, rgba(117, 64, 206, 0.08) 0%, transparent 50%),
            radial-gradient(circle at 85% 80%, rgba(117, 64, 206, 0.06) 0%, transparent 50%);
        animation: breathe 15s ease-in-out infinite;
        pointer-events: none;
        z-index: 0;
    }
    
    @keyframes breathe {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    
    /* Hide Streamlit Branding */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* === TEXT INPUT === */
    .stTextInput > div > div > input {
        background: rgba(30, 25, 50, 0.6) !important;
        backdrop-filter: blur(20px) !important;
        border: 2px solid rgba(117, 64, 206, 0.3) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        font-size: 1.1rem !important;
        padding: 1.2rem 1.5rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        font-family: 'Poppins', sans-serif !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #7540ce !important;
        box-shadow: 0 0 0 4px rgba(117, 64, 206, 0.2), 0 8px 30px rgba(117, 64, 206, 0.3) !important;
        transform: translateY(-2px);
    }
    
    .stTextInput > div > div > input::placeholder {
        color: rgba(255, 255, 255, 0.4) !important;
    }
    
    /* === BUTTON STYLING === */
    .stButton > button {
        background: linear-gradient(135deg, #7540ce 0%, #9d67e8 100%) !important;
        color: #ffffff !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 1rem 2.5rem !important;
        cursor: pointer !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 8px 24px rgba(117, 64, 206, 0.4) !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #8654d9 0%, #a876f3 100%) !important;
        transform: translateY(-3px);
        box-shadow: 0 12px 32px rgba(117, 64, 206, 0.6) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* === RADIO BUTTONS (Model Selection Cards) === */
    .stRadio {
        background: transparent !important;
    }
    
    .stRadio > div {
        display: flex;
        gap: 1.5rem;
        justify-content: center;
        flex-wrap: wrap;
        background: transparent !important;
    }
    
    .stRadio > div > label {
        background: rgba(30, 25, 50, 0.5) !important;
        backdrop-filter: blur(20px) !important;
        border: 2px solid rgba(117, 64, 206, 0.2) !important;
        border-radius: 16px !important;
        padding: 1.8rem 2.2rem !important;
        cursor: pointer !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        font-family: 'Space Grotesk', sans-serif !important;
        color: rgba(255, 255, 255, 0.9) !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
        min-width: 200px;
        text-align: center;
    }
    
    .stRadio > div > label:hover {
        transform: translateY(-4px) scale(1.02);
        border-color: #7540ce !important;
        box-shadow: 0 8px 30px rgba(117, 64, 206, 0.4) !important;
        background: rgba(117, 64, 206, 0.15) !important;
    }
    
    .stRadio > div > label[data-baseweb="radio"] > div:first-child {
        border-color: #7540ce !important;
        background: rgba(117, 64, 206, 0.2) !important;
    }
    
    .stRadio > div > label[data-baseweb="radio"]:has(input:checked) {
        border-color: #7540ce !important;
        box-shadow: 0 0 0 4px rgba(117, 64, 206, 0.3), 0 8px 30px rgba(117, 64, 206, 0.5) !important;
        background: rgba(117, 64, 206, 0.2) !important;
        transform: scale(1.05);
    }
    
    /* === INFO/SUCCESS MESSAGES === */
    .stSuccess, .stInfo {
        background: rgba(30, 25, 50, 0.6) !important;
        backdrop-filter: blur(20px) !important;
        border-left: 4px solid #7540ce !important;
        border-radius: 12px !important;
        color: rgba(255, 255, 255, 0.9) !important;
        padding: 1rem 1.5rem !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
    }
    
    /* === EXPANDER === */
    .streamlit-expanderHeader {
        background: rgba(30, 25, 50, 0.6) !important;
        backdrop-filter: blur(20px) !important;
        border-radius: 12px !important;
        color: rgba(255, 255, 255, 0.9) !important;
        font-family: 'Space Grotesk', sans-serif !important;
        border: 1px solid rgba(117, 64, 206, 0.3) !important;
        padding: 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #7540ce !important;
        box-shadow: 0 4px 20px rgba(117, 64, 206, 0.3) !important;
    }
    
    /* === SPINNER === */
    .stSpinner > div {
        border-top-color: #7540ce !important;
        border-right-color: rgba(117, 64, 206, 0.3) !important;
    }
    
    /* === MARKDOWN TEXT === */
    .stMarkdown {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }
    
    p {
        color: rgba(255, 255, 255, 0.8) !important;
    }
    
    /* === CUSTOM LOADER === */
    .custom-loader {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 0.8rem;
        padding: 2rem;
    }
    
    .custom-loader span {
        width: 14px;
        height: 14px;
        background: #7540ce;
        border-radius: 50%;
        animation: bounce 1.4s infinite ease-in-out both;
    }
    
    .custom-loader span:nth-child(1) { animation-delay: -0.32s; }
    .custom-loader span:nth-child(2) { animation-delay: -0.16s; }
    
    @keyframes bounce {
        0%, 80%, 100% { transform: scale(0); opacity: 0.5; }
        40% { transform: scale(1); opacity: 1; }
    }
    
    /* === STATUS INDICATOR === */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s ease-in-out infinite;
    }
    
    .status-active {
        background: #4ade80;
        box-shadow: 0 0 10px rgba(74, 222, 128, 0.5);
    }
    
    .status-inactive {
        background: #ef4444;
        box-shadow: 0 0 10px rgba(239, 68, 68, 0.5);
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .status-card {
        background: rgba(30, 25, 50, 0.6);
        backdrop-filter: blur(20px);
        border: 2px solid rgba(117, 64, 206, 0.3);
        border-radius: 16px;
        padding: 1rem 1.5rem;
        margin: 1rem auto;
        max-width: 600px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    
    /* === LOGO STYLING === */
    .logo-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    .logo-container img {
        max-width: 180px;
        height: auto;
        filter: drop-shadow(0 4px 20px rgba(117, 64, 206, 0.4));
    }
    
    /* === RESPONSIVE === */
    @media (max-width: 768px) {
        .stRadio > div > label {
            min-width: 150px;
            padding: 1.2rem 1.5rem !important;
        }
        .logo-container img {
            max-width: 140px;
        }
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
    
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=inputs.shape[1] + (60 if is_finetuned else 40),
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

def get_base64_image(image_path):
    """Convert image to base64 for embedding"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

# Main UI
def main():
    # Load both models
    with st.spinner("Loading AI models..."):
        finetuned_model, finetuned_tokenizer, finetuned_info = load_finetuned_model()
        base_model, base_tokenizer, base_info = load_base_model()
    
    # Get logo
    logo_path = "BluffifyLogo.png"
    logo_base64 = get_base64_image(logo_path)
    
    # Professional Header with Logo and Brand
    if logo_base64:
        st.markdown(f"""
        <div style='text-align: center; padding: 2rem 0 1.5rem 0;'>
            <div class='logo-container'>
                <img src='data:image/png;base64,{logo_base64}' alt='Bluffify Logo'>
            </div>
            <h1 style='font-family: "Space Grotesk", sans-serif; font-size: 3.5rem; font-weight: 700;
                       color: #ffffff; margin: 0.5rem 0 0 0; letter-spacing: -1px;'>
                Bluffify
            </h1>
            <div style='height: 4px; width: 200px; margin: 1.5rem auto;
                        background: linear-gradient(90deg, transparent, #7540ce, transparent);
                        border-radius: 2px;'></div>
            <p style='font-size: 1.2rem; color: rgba(255, 255, 255, 0.7); margin: 0;
                      font-weight: 300; max-width: 600px; margin: 0 auto; line-height: 1.6;'>
                Generate hilariously absurd fake news headlines using AI-powered language models
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='text-align: center; padding: 3rem 0 2rem 0;'>
            <h1 style='font-family: "Space Grotesk", sans-serif; font-size: 3.5rem; font-weight: 700;
                       color: #ffffff; margin: 0; letter-spacing: -1px;'>
                🎭 Bluffify
            </h1>
            <div style='height: 4px; width: 200px; margin: 1.5rem auto;
                        background: linear-gradient(90deg, transparent, #7540ce, transparent);
                        border-radius: 2px;'></div>
            <p style='font-size: 1.2rem; color: rgba(255, 255, 255, 0.7); margin: 0;
                      font-weight: 300; max-width: 600px; margin: 0 auto; line-height: 1.6;'>
                Generate hilariously absurd fake news headlines using AI-powered language models
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Model Status Indicator
    finetuned_status = "active" if finetuned_info else "inactive"
    base_status = "active" if base_info else "inactive"
    
    st.markdown(f"""
    <div class='status-card'>
        <div style='display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap; gap: 1rem;'>
            <div style='display: flex; align-items: center;'>
                <span class='status-indicator status-{finetuned_status}'></span>
                <span style='font-family: "Space Grotesk", sans-serif; color: rgba(255, 255, 255, 0.9); font-weight: 500;'>
                    Fine-Tuned Model: <strong style='color: {'#4ade80' if finetuned_info else '#ef4444'};'>{'Active' if finetuned_info else 'Inactive'}</strong>
                </span>
            </div>
            <div style='display: flex; align-items: center;'>
                <span class='status-indicator status-{base_status}'></span>
                <span style='font-family: "Space Grotesk", sans-serif; color: rgba(255, 255, 255, 0.9); font-weight: 500;'>
                    Base GPT-2 Model: <strong style='color: {'#4ade80' if base_info else '#ef4444'};'>{'Active' if base_info else 'Inactive'}</strong>
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Model Selection Section with Card Icons
    st.markdown("""
    <h2 style='font-family: "Space Grotesk", sans-serif; font-size: 1.5rem; font-weight: 600;
               color: #7540ce; text-align: center; margin: 2rem 0 1.5rem 0;'>
        Choose Your Model
    </h2>
    """, unsafe_allow_html=True)
    
    if finetuned_info:
        mode = st.radio(
            "Choose generation mode:",
            ["Fine-Tuned Model", "Base Model", "Compare Both"],
            horizontal=True
        )
    else:
        mode = "Base Model"
        st.info("Fine-tuned model not found. Using base model only.")
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Input Section with Professional Styling
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <h3 style='font-family: "Space Grotesk", sans-serif; font-size: 1.3rem; font-weight: 600;
                   color: rgba(255, 255, 255, 0.9); text-align: center; margin-bottom: 1rem;'>
            Enter Your Topic
        </h3>
        """, unsafe_allow_html=True)
        
        topic = st.text_input(
            "Topic",
            placeholder="e.g., pizza, robots, cryptocurrency, exams...",
            label_visibility="collapsed",
            key="topic_input"
        )
        
        st.markdown("""
        <p style='text-align: center; color: rgba(255, 255, 255, 0.5); font-size: 0.9rem;
                  margin: 0.5rem 0 1.5rem 0;'>
            Try: pizza, aliens, exams, crypto, politicians
        </p>
        """, unsafe_allow_html=True)
        
        generate_btn = st.button("Generate Headline", type="primary")
    
    # Generation Section
    if generate_btn:
        if topic.strip():
            
            # Compare mode - Professional Side-by-Side Cards
            if mode == "Compare Both" and finetuned_info:
                # Create a placeholder for the loader that we can clear
                loader_placeholder = st.empty()
                
                # Show loader
                with loader_placeholder.container():
                    st.markdown("""
                    <div class="custom-loader">
                        <span></span><span></span><span></span>
                    </div>
                    <p style='text-align: center; color: rgba(255, 255, 255, 0.6); margin-top: 1rem;'>
                        Generating with both models...
                    </p>
                    """, unsafe_allow_html=True)
                
                # Generate headlines
                headline_ft = generate_headline(topic, finetuned_model, finetuned_tokenizer, finetuned_info)
                headline_base = generate_headline(topic, base_model, base_tokenizer, base_info)
                
                # Clear the loader
                loader_placeholder.empty()
                
                col_ft, col_base = st.columns(2, gap="large")
                
                # Fine-tuned Model Result
                with col_ft:
                    st.markdown(f"""
                    <div style='background: rgba(30, 25, 50, 0.6); backdrop-filter: blur(30px);
                                border: 2px solid rgba(117, 64, 206, 0.4); border-radius: 16px;
                                padding: 2rem; box-shadow: 0 8px 32px rgba(117, 64, 206, 0.3);
                                transition: transform 0.3s ease; animation: fadeInUp 0.6s ease;'>
                        <div style='display: inline-block; background: linear-gradient(135deg, #7540ce, #9d67e8);
                                    padding: 0.5rem 1.2rem; border-radius: 8px; font-size: 0.85rem;
                                    font-weight: 600; text-transform: uppercase; letter-spacing: 1px;
                                    margin-bottom: 1.5rem; box-shadow: 0 4px 15px rgba(117, 64, 206, 0.4);'>
                            Fine-Tuned Model
                        </div>
                        <div style='font-family: "Poppins", sans-serif; font-size: 1.35rem;
                                    line-height: 1.7; color: #ffffff; font-weight: 500;'>
                            {headline_ft}
                        </div>
                        <div style='margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid rgba(255, 255, 255, 0.1);
                                    font-size: 0.85rem; color: rgba(255, 255, 255, 0.6);'>
                            Trained on 8,000 headlines | 124M parameters
                        </div>
                    </div>
                    <style>
                        @keyframes fadeInUp {{
                            from {{ opacity: 0; transform: translateY(30px); }}
                            to {{ opacity: 1; transform: translateY(0); }}
                        }}
                    </style>
                    """, unsafe_allow_html=True)
                
                # Base Model Result
                with col_base:
                    st.markdown(f"""
                    <div style='background: rgba(30, 25, 50, 0.6); backdrop-filter: blur(30px);
                                border: 2px solid rgba(100, 100, 120, 0.3); border-radius: 16px;
                                padding: 2rem; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                                transition: transform 0.3s ease; animation: fadeInUp 0.6s ease 0.1s both;'>
                        <div style='display: inline-block; background: linear-gradient(135deg, #6c757d, #8a929a);
                                    padding: 0.5rem 1.2rem; border-radius: 8px; font-size: 0.85rem;
                                    font-weight: 600; text-transform: uppercase; letter-spacing: 1px;
                                    margin-bottom: 1.5rem; box-shadow: 0 4px 15px rgba(100, 100, 120, 0.3);'>
                            Base Model
                        </div>
                        <div style='font-family: "Poppins", sans-serif; font-size: 1.35rem;
                                    line-height: 1.7; color: #ffffff; font-weight: 500;'>
                            {headline_base}
                        </div>
                        <div style='margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid rgba(255, 255, 255, 0.1);
                                    font-size: 0.85rem; color: rgba(255, 255, 255, 0.6);'>
                            GPT-2 Large | 774M parameters
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Single Model - Professional Result Card
            else:
                if mode == "Fine-Tuned Model" and finetuned_info:
                    model, tokenizer, model_info = finetuned_model, finetuned_tokenizer, finetuned_info
                    chip_bg = "linear-gradient(135deg, #7540ce, #9d67e8)"
                    border_color = "rgba(117, 64, 206, 0.4)"
                    shadow_color = "rgba(117, 64, 206, 0.3)"
                    model_details = "Trained on 8,000 headlines | 124M parameters"
                else:
                    model, tokenizer, model_info = base_model, base_tokenizer, base_info
                    chip_bg = "linear-gradient(135deg, #6c757d, #8a929a)"
                    border_color = "rgba(100, 100, 120, 0.3)"
                    shadow_color = "rgba(0, 0, 0, 0.3)"
                    model_details = "GPT-2 Large | 774M parameters"
                
                # Create a placeholder for the loader
                loader_placeholder = st.empty()
                
                # Show loader
                with loader_placeholder.container():
                    st.markdown("""
                    <div class="custom-loader">
                        <span></span><span></span><span></span>
                    </div>
                    <p style='text-align: center; color: rgba(255, 255, 255, 0.6); margin-top: 1rem;'>
                        Generating your headline...
                    </p>
                    """, unsafe_allow_html=True)
                
                # Generate headline
                headline = generate_headline(topic, model, tokenizer, model_info)
                
                # Clear the loader
                loader_placeholder.empty()
                
                col1, col2, col3 = st.columns([0.5, 3, 0.5])
                with col2:
                    st.markdown(f"""
                    <div style='background: rgba(30, 25, 50, 0.6); backdrop-filter: blur(30px);
                                border: 2px solid {border_color}; border-radius: 16px;
                                padding: 2.5rem; box-shadow: 0 8px 32px {shadow_color};
                                margin: 2rem 0; animation: fadeInUp 0.6s ease;'>
                        <div style='text-align: center; margin-bottom: 2rem;'>
                            <div style='display: inline-block; background: {chip_bg};
                                        padding: 0.6rem 1.5rem; border-radius: 8px; font-size: 0.9rem;
                                        font-weight: 600; text-transform: uppercase; letter-spacing: 1px;
                                        box-shadow: 0 4px 15px {shadow_color};'>
                                {model_info['name']}
                            </div>
                        </div>
                        <div style='font-family: "Poppins", sans-serif; font-size: 1.8rem;
                                    line-height: 1.7; color: #ffffff; font-weight: 500;
                                    text-align: center; padding: 1rem 0;'>
                            {headline}
                        </div>
                        <div style='margin-top: 2rem; padding-top: 1.5rem; border-top: 1px solid rgba(255, 255, 255, 0.1);
                                    text-align: center; font-size: 0.9rem; color: rgba(255, 255, 255, 0.6);'>
                            {model_details}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.info("💡 Note: This is AI-generated fiction for entertainment purposes only.")
            
        else:
            st.error("⚠️ Please enter a topic to generate a headline.")
    
    # Model Information Panel
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    with st.expander("📊 Model Information & Technical Details"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style='background: rgba(117, 64, 206, 0.1); border: 1px solid rgba(117, 64, 206, 0.3);
                        border-radius: 12px; padding: 1.5rem; text-align: center;'>
                <h4 style='color: #7540ce; margin: 0 0 1rem 0; font-size: 1.1rem;'>Fine-Tuned Model</h4>
                <p style='color: rgba(255, 255, 255, 0.8); margin: 0.5rem 0; font-size: 0.95rem;'>
                    <strong>Architecture:</strong> GPT-2 Base<br>
                    <strong>Parameters:</strong> 124M<br>
                    <strong>Training Data:</strong> 8,000 headlines<br>
                    <strong>Epochs:</strong> 3<br>
                    <strong>Specialty:</strong> Authentic fake news style
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='background: rgba(100, 100, 120, 0.1); border: 1px solid rgba(100, 100, 120, 0.3);
                        border-radius: 12px; padding: 1.5rem; text-align: center;'>
                <h4 style='color: #8a929a; margin: 0 0 1rem 0; font-size: 1.1rem;'>Base Model</h4>
                <p style='color: rgba(255, 255, 255, 0.8); margin: 0.5rem 0; font-size: 0.95rem;'>
                    <strong>Architecture:</strong> GPT-2 Large<br>
                    <strong>Parameters:</strong> 774M<br>
                    <strong>Training Data:</strong> General web text<br>
                    <strong>Source:</strong> HuggingFace<br>
                    <strong>Specialty:</strong> Creative generation
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style='background: rgba(117, 64, 206, 0.08); border: 1px solid rgba(117, 64, 206, 0.2);
                        border-radius: 12px; padding: 1.5rem; text-align: center;'>
                <h4 style='color: #9d67e8; margin: 0 0 1rem 0; font-size: 1.1rem;'>Key Differences</h4>
                <p style='color: rgba(255, 255, 255, 0.8); margin: 0.5rem 0; font-size: 0.95rem;'>
                    <strong>Fine-Tuned:</strong> Specialized, authentic<br>
                    <strong>Base:</strong> General, creative<br>
                    <strong>Best for authenticity:</strong> Fine-Tuned<br>
                    <strong>Best for variety:</strong> Base<br>
                    <strong>Speed:</strong> Similar
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background: rgba(30, 25, 50, 0.4); border-radius: 12px; padding: 1.5rem;
                    border: 1px solid rgba(117, 64, 206, 0.2);'>
            <h4 style='color: #7540ce; margin: 0 0 1rem 0;'>How It Works</h4>
            <p style='color: rgba(255, 255, 255, 0.8); line-height: 1.8; margin: 0;'>
                This AI-powered application uses transformer-based language models to generate satirical fake news headlines.
                The <strong>Fine-Tuned Model</strong> has been specifically trained on 8,000 real fake news headlines to understand
                and replicate authentic fake news patterns. The <strong>Base Model</strong> uses general language understanding
                for more creative outputs. The <strong>Compare Both</strong> mode lets you see the difference side-by-side.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Professional Footer
    st.markdown("""
    <div style='margin-top: 5rem; padding: 2.5rem 0; text-align: center;
                border-top: 1px solid rgba(117, 64, 206, 0.2);'>
        <p style='font-size: 1rem; color: rgba(255, 255, 255, 0.7); margin: 0.5rem 0;
                  font-weight: 300;'>
            🎭 Built with Streamlit & HuggingFace Transformers
        </p>
        <p style='font-size: 0.9rem; color: rgba(255, 255, 255, 0.5); margin: 1rem 0;'>
            <a href='https://github.com/Hissan-2002/fake-news-headline-generator'
               style='color: #7540ce; text-decoration: none; transition: color 0.3s ease;
                      font-weight: 500;'
               onmouseover='this.style.color=\"#9d67e8\"'
               onmouseout='this.style.color=\"#7540ce\"'>
                View Bluffify on GitHub
            </a>
        </p>
        <p style='font-size: 0.85rem; color: rgba(255, 255, 255, 0.4); margin: 0.5rem 0;'>
            All generated headlines are AI-created fiction for entertainment purposes only.
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
