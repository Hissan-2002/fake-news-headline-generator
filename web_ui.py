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

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    /* === GLOBAL THEME & CONTAINER === */
    .stApp {
        background: linear-gradient(135deg, #0f0c1e 0%, #1a1635 100%);
        background-attachment: fixed;
        font-family: 'Poppins', sans-serif;
    }

    /* Center the main content area and limit width on huge screens */
    .block-container {
        max-width: 1200px;
        padding-top: 3rem !important;
        padding-bottom: 5rem !important;
        margin: 0 auto;
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
    
    /* === TEXT INPUT STYLING === */
    /* Center the input widgets */
    div[data-testid="stTextInput"] {
        text-align: center;
        margin: 0 auto;
    }

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
        text-align: center; /* Center text inside input */
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
    .stButton {
        display: flex;
        justify-content: center;
    }

    .stButton > button {
        background: linear-gradient(135deg, #7540ce 0%, #9d67e8 100%) !important;
        color: #ffffff !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 1rem 3rem !important; /* Wider padding */
        cursor: pointer !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 8px 24px rgba(117, 64, 206, 0.4) !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        min-width: 250px; /* Ensure button isn't too small */
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
        margin-top: 1rem;
    }
    
    .stRadio > div[role="radiogroup"] {
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
        padding: 1.5rem 2rem !important;
        cursor: pointer !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        font-family: 'Space Grotesk', sans-serif !important;
        color: rgba(255, 255, 255, 0.9) !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
        min-width: 180px;
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    
    .stRadio > div > label:hover {
        transform: translateY(-4px) scale(1.02);
        border-color: #7540ce !important;
        box-shadow: 0 8px 30px rgba(117, 64, 206, 0.4) !important;
        background: rgba(117, 64, 206, 0.15) !important;
    }
    
    /* Selected State */
    .stRadio > div > label[data-baseweb="radio"] > div:first-child {
        margin-bottom: 8px; /* Spacing between circle and text */
        border-color: #7540ce !important;
        background: rgba(117, 64, 206, 0.2) !important;
    }
    
    /* === INFO/SUCCESS MESSAGES === */
    .stSuccess, .stInfo, .stError {
        background: rgba(30, 25, 50, 0.6) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: rgba(255, 255, 255, 0.9) !important;
        padding: 1rem 1.5rem !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
    }
    .stSuccess { border-left: 4px solid #4ade80 !important; }
    .stInfo { border-left: 4px solid #7540ce !important; }
    .stError { border-left: 4px solid #ef4444 !important; }
    
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
    
    .streamlit-expanderContent {
        background: rgba(30, 25, 50, 0.4) !important;
        border-radius: 0 0 12px 12px !important;
        border: 1px solid rgba(117, 64, 206, 0.2) !important;
        border-top: none !important;
    }
    
    /* === SPINNER === */
    .stSpinner > div {
        border-top-color: #7540ce !important;
        border-right-color: rgba(117, 64, 206, 0.3) !important;
    }
    
    /* === TYPOGRAPHY === */
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
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
        vertical-align: middle;
    }
    
    .status-active {
        background: #4ade80;
        box-shadow: 0 0 8px rgba(74, 222, 128, 0.6);
        animation: pulse 2s ease-in-out infinite;
    }
    
    .status-inactive {
        background: #ef4444;
        box-shadow: 0 0 8px rgba(239, 68, 68, 0.6);
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
        margin: 1.5rem auto 2.5rem auto;
        max-width: 700px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    
    /* === LOGO STYLING === */
    .logo-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    
    .logo-container img {
        max-width: 150px;
        height: auto;
        filter: drop-shadow(0 4px 20px rgba(117, 64, 206, 0.4));
        transition: transform 0.3s ease;
    }
    
    .logo-container img:hover {
        transform: scale(1.05);
    }
    
    /* === RESPONSIVE === */
    @media (max-width: 768px) {
        .stRadio > div > label {
            min-width: 100%;
            margin-bottom: 0.5rem;
        }
        .logo-container img {
            max-width: 120px;
        }
        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Cache models separately
@st.cache_resource
def load_old_finetuned_model():
    """Load old full fine-tuned model"""
    path = "./fine_tuned_model/final"
    if os.path.exists(path):
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path)
        tokenizer.pad_token = tokenizer.eos_token
        
        model_info = {
            "name": "Old Fine-Tuned",
            "is_finetuned": True,
            "model_type": "old"
        }
        return model, tokenizer, model_info
    return None, None, None

@st.cache_resource
def load_lora_model():
    """Load new LoRA fine-tuned model (merged)"""
    path = "./fine_tuned_model_lora/final_merged"
    if os.path.exists(path):
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path)
        tokenizer.pad_token = tokenizer.eos_token
        
        model_info = {
            "name": "LoRA Fine-Tuned",
            "is_finetuned": True,
            "model_type": "lora"
        }
        return model, tokenizer, model_info
    return None, None, None

@st.cache_resource
def load_base_model():
    """Load GPT-2 base model"""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    model_info = {
        "name": "Base GPT-2",
        "is_finetuned": False,
        "model_type": "base"
    }
    
    return model, tokenizer, model_info

def generate_headline(topic, model, tokenizer, model_info):
    """Generate funny and humorous fake news headline"""
    model_type = model_info.get("model_type", "base")
    
    # Try multiple generation attempts and pick the funniest one
    best_headline = None
    best_score = -1
    
    for attempt in range(3):  # Try 3 times for better quality
        # Use humorous prompts
        prompts = [
            f"Write a hilarious satirical fake news headline about {topic.strip()}:",
            f"Create an absurd and funny headline about {topic.strip()}:",
            f"Generate a ridiculous breaking news headline about {topic.strip()}:"
        ]
        prompt = prompts[attempt % len(prompts)]
        
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + 35,
                min_length=inputs.shape[1] + 10,
                do_sample=True,
                temperature=0.9 + (attempt * 0.05),  # Higher temp for creativity
                top_p=0.92,
                top_k=50,
                repetition_penalty=1.8,
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        headline = full_output.replace(prompt, "").strip()
        
        # Aggressive cleaning
        import re
        
        # Remove everything in brackets, parentheses, quotes
        headline = re.sub(r'\[.*?\]|\(.*?\)|".*?"', '', headline)
        
        # Remove URLs and web-related text
        headline = re.sub(r'http\S+|www\.\S+', '', headline)
        headline = re.sub(r'\.(com|org|net|gov)\S*', '', headline)
        
        # Remove common garbage words/phrases
        garbage = ['VIDEO', 'CLICK', 'IMAGE', 'PHOTO', 'Watch', 'Here', 'Link', 
                   'Un-Called', 'Express Has', 'Shows Who', 'Free Expression',
                   'Bill Of Rights', 'Paid For']
        for word in garbage:
            headline = re.sub(re.escape(word), '', headline, flags=re.IGNORECASE)
        
        # Remove special characters and clean up
        headline = re.sub(r'[…–—]', '', headline)
        headline = re.sub(r'\s+', ' ', headline)
        
        # Clean prompt remnants
        headline = re.sub(r'^(Write|Create|Generate|Headline|Breaking|Exclusive|Satirical|Funny|Ridiculous|Absurd)[:,]?\s*', '', headline, flags=re.IGNORECASE)
        
        # Take only first complete sentence
        sentences = re.split(r'[.!?]', headline)
        headline = sentences[0].strip() if sentences else headline.strip()
        
        # Remove leading punctuation
        headline = headline.lstrip(':;-–—"\' ')
        
        # Score for humor and quality
        score = 0
        topic_lower = topic.lower()
        headline_lower = headline.lower()
        
        # Topic relevance is key
        if topic_lower in headline_lower:
            score += 20
        
        # Reward absurd/funny words
        funny_words = ['declares', 'reveals', 'scientists', 'discovers', 'study', 'experts',
                       'shocked', 'horrified', 'demands', 'causes', 'accidentally', 'secretly',
                       'refuses', 'admits', 'claims', 'warns', 'predicts', 'announces']
        score += sum(5 for word in funny_words if word in headline_lower)
        
        # Good length
        if 30 < len(headline) < 120:
            score += 15
        
        # Has capitalization
        if any(char.isupper() for char in headline):
            score += 5
        
        # No garbage text
        if not any(bad in headline_lower for bad in ['http', 'www', 'click', 'watch', 'video']):
            score += 10
        
        # Has numbers (funnier)
        if any(char.isdigit() for char in headline):
            score += 3
        
        if score > best_score and len(headline) > 15:
            best_score = score
            best_headline = headline
    
    # Use best headline or create a funny fallback
    if best_headline and len(best_headline) > 15:
        headline = best_headline
    else:
        # Hilarious topic-specific fallback headlines
        funny_templates = [
            f"Scientists Discover {topic.title()} Can Cure Monday Blues",
            f"Breaking: {topic.title()} Declared Eighth Wonder of the World",
            f"{topic.title()} Causes Mass Confusion at UN Meeting",
            f"Study Reveals {topic.title()} is 99% Magic, 1% Mystery",
            f"Experts Warn: Too Much {topic.title()} May Cause Spontaneous Dancing",
            f"World Leaders Gather to Debate Proper Way to Enjoy {topic.title()}",
            f"{topic.title()} Accidentally Solves Climate Change, Scientists Baffled",
            f"Breaking: {topic.title()} Voted 'Most Likely to Succeed' by Aliens",
            f"Local Man Refuses to Share {topic.title()}, Nation Divided",
            f"Study: 10 Out of 10 Cats Prefer {topic.title()} Over World Peace"
        ]
        import random
        headline = random.choice(funny_templates)
    
    # Final cleanup and formatting
    headline = headline.strip()
    
    # Capitalize first letter
    if headline and headline[0].islower():
        headline = headline[0].upper() + headline[1:]
    
    # Add exclamation for comedy effect if no punctuation
    if headline and not headline.endswith(('.', '!', '?')):
        # Use exclamation for funnier effect
        headline += "!"
    
    # Final validation
    if len(headline) < 20 or len(headline) > 150:
        return f"Breaking: {topic.title()} Causes Unprecedented Outbreak of Uncontrollable Giggles!"
    
    return headline

def get_base64_image(image_path):
    """Convert image to base64 for embedding"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

# Main UI
def main():
    # Load all three models
    with st.spinner("Loading AI models..."):
        old_model, old_tokenizer, old_info = load_old_finetuned_model()
        lora_model, lora_tokenizer, lora_info = load_lora_model()
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
            <div style='height: 4px; width: 120px; margin: 1rem auto;
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
        <div style='text-align: center; padding: 2rem 0 1rem 0;'>
            <h1 style='font-family: "Space Grotesk", sans-serif; font-size: 4rem; font-weight: 700;
                        color: #ffffff; margin: 0; letter-spacing: -2px; text-shadow: 0 4px 10px rgba(117, 64, 206, 0.5);'>
                🎭 Bluffify
            </h1>
            <div style='height: 4px; width: 120px; margin: 1rem auto;
                        background: linear-gradient(90deg, transparent, #7540ce, transparent);
                        border-radius: 2px;'></div>
            <p style='font-size: 1.2rem; color: rgba(255, 255, 255, 0.7); margin: 0;
                      font-weight: 300; max-width: 600px; margin: 0 auto; line-height: 1.6;'>
                Generate hilariously absurd fake news headlines using AI-powered language models
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Model Status Indicator - Show all 3 models
    old_status = "active" if old_info else "inactive"
    lora_status = "active" if lora_info else "inactive"
    base_status = "active" if base_info else "inactive"
    
    st.markdown(f"""
    <div class='status-card'>
        <div style='display: flex; justify-content: space-evenly; align-items: center; flex-wrap: wrap; gap: 1.5rem;'>
            <div style='display: flex; align-items: center;'>
                <span class='status-indicator status-{old_status}'></span>
                <span style='font-family: "Space Grotesk", sans-serif; color: rgba(255, 255, 255, 0.9); font-weight: 500; font-size: 0.9rem;'>
                    Old Fine-Tuned: <strong style='color: {'#4ade80' if old_info else '#ef4444'};'>{'Ready' if old_info else 'Missing'}</strong>
                </span>
            </div>
            <div style='display: flex; align-items: center;'>
                <span class='status-indicator status-{lora_status}'></span>
                <span style='font-family: "Space Grotesk", sans-serif; color: rgba(255, 255, 255, 0.9); font-weight: 500; font-size: 0.9rem;'>
                    LoRA Fine-Tuned: <strong style='color: {'#4ade80' if lora_info else '#ef4444'};'>{'Ready' if lora_info else 'Missing'}</strong>
                </span>
            </div>
            <div style='display: flex; align-items: center;'>
                <span class='status-indicator status-{base_status}'></span>
                <span style='font-family: "Space Grotesk", sans-serif; color: rgba(255, 255, 255, 0.9); font-weight: 500; font-size: 0.9rem;'>
                    Base GPT-2: <strong style='color: {'#4ade80' if base_info else '#ef4444'};'>{'Ready' if base_info else 'Missing'}</strong>
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Selection Section with Card Icons
    st.markdown("""
    <h2 style='font-family: "Space Grotesk", sans-serif; font-size: 1.5rem; font-weight: 600;
               color: #7540ce; text-align: center; margin: 3rem 0 1rem 0; letter-spacing: -0.5px;'>
        SELECT YOUR MODEL
    </h2>
    """, unsafe_allow_html=True)
    
    # Build available options based on loaded models
    available_options = []
    if old_info:
        available_options.append("Old Fine-Tuned")
    if lora_info:
        available_options.append("LoRA Fine-Tuned (New)")
    available_options.append("Base GPT-2")
    
    # Add compare option if we have at least 2 models
    available_model_count = sum([bool(old_info), bool(lora_info), True])  # Base always available
    if available_model_count >= 2:
        available_options.append("Compare All Models")
    
    # Centered container for radio buttons
    col_rad1, col_rad2, col_rad3 = st.columns([1, 4, 1])
    with col_rad2:
        if len(available_options) > 1:
            mode = st.radio(
                "Choose generation mode:",
                available_options,
                horizontal=True,
                index=1 if lora_info else 0,  # Default to LoRA if available
                label_visibility="collapsed"
            )
        else:
            mode = available_options[0]
            st.info(f"Only {mode} is available.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Input Section with Professional Styling - CENTERED ALIGNMENT
    # Adjusted columns for better centering
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <h3 style='font-family: "Space Grotesk", sans-serif; font-size: 1.3rem; font-weight: 600;
                    color: rgba(255, 255, 255, 0.9); text-align: center; margin-bottom: 0.5rem;'>
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
        <p style='text-align: center; color: rgba(255, 255, 255, 0.5); font-size: 0.85rem;
                  margin: 0.5rem 0 1.5rem 0;'>
            Try: <em>pizza, aliens, exams, crypto, politicians</em>
        </p>
        """, unsafe_allow_html=True)
        
        # Centered button via CSS and container
        generate_btn = st.button("Generate Headline", type="primary", use_container_width=True)
    
    # Generation Section
    if generate_btn:
        if topic.strip():
            
            # Compare All Models - Show all available models side-by-side
            if mode == "Compare All Models":
                # Create a placeholder for the loader that we can clear
                loader_placeholder = st.empty()
                
                # Show loader
                with loader_placeholder.container():
                    st.markdown("""
                    <div class="custom-loader">
                        <span></span><span></span><span></span>
                    </div>
                    <p style='text-align: center; color: rgba(255, 255, 255, 0.6); margin-top: 1rem;'>
                        Generating with all available models...
                    </p>
                    """, unsafe_allow_html=True)
                
                # Generate headlines for all available models
                headlines = []
                
                if old_info:
                    headline_old = generate_headline(topic, old_model, old_tokenizer, old_info)
                    headlines.append(("Old Fine-Tuned", headline_old, "linear-gradient(135deg, #e67e22, #f39c12)", "rgba(230, 126, 34, 0.3)", "Full Fine-Tuned | 3 epochs"))
                
                if lora_info:
                    headline_lora = generate_headline(topic, lora_model, lora_tokenizer, lora_info)
                    headlines.append(("LoRA Fine-Tuned", headline_lora, "linear-gradient(135deg, #7540ce, #9d67e8)", "rgba(117, 64, 206, 0.3)", "LoRA | 5 epochs | Context-Aware"))
                
                headline_base = generate_headline(topic, base_model, base_tokenizer, base_info)
                headlines.append(("Base GPT-2", headline_base, "linear-gradient(135deg, #6c757d, #8a929a)", "rgba(100, 100, 120, 0.3)", "124M parameters"))
                
                # Clear the loader
                loader_placeholder.empty()
                
                st.markdown("<br>", unsafe_allow_html=True)

                # Display all models side-by-side
                cols = st.columns(len(headlines), gap="medium")
                
                for idx, (name, headline, chip_bg, shadow_color, details) in enumerate(headlines):
                    with cols[idx]:
                        st.markdown(f"""
                        <div style='background: rgba(30, 25, 50, 0.6); backdrop-filter: blur(30px);
                                    border: 2px solid {shadow_color.replace('0.3', '0.4')}; border-radius: 16px;
                                    padding: 2rem; box-shadow: 0 8px 32px {shadow_color};
                                    transition: transform 0.3s ease; animation: fadeInUp 0.6s ease {idx * 0.1}s both;
                                    height: 100%; display: flex; flex-direction: column; justify-content: space-between;'>
                            <div>
                                <div style='display: inline-block; background: {chip_bg};
                                            padding: 0.5rem 1rem; border-radius: 8px; font-size: 0.8rem;
                                            font-weight: 600; text-transform: uppercase; letter-spacing: 1px;
                                            margin-bottom: 1.5rem; box-shadow: 0 4px 15px {shadow_color}; color: white;'>
                                    {name}
                                </div>
                                <div style='font-family: "Poppins", sans-serif; font-size: 1.15rem;
                                            line-height: 1.6; color: #ffffff; font-weight: 500;
                                            min-height: 100px;'>
                                    {headline}
                                </div>
                            </div>
                            <div style='margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid rgba(255, 255, 255, 0.1);
                                        font-size: 0.75rem; color: rgba(255, 255, 255, 0.5); font-family: "Space Grotesk", sans-serif;'>
                                {details}
                            </div>
                        </div>
                        <style>
                            @keyframes fadeInUp {{
                                from {{ opacity: 0; transform: translateY(30px); }}
                                to {{ opacity: 1; transform: translateY(0); }}
                            }}
                        </style>
                        """, unsafe_allow_html=True)
            
            # Single Model - Professional Result Card
            else:
                if mode == "LoRA Fine-Tuned (New)" and lora_info:
                    model, tokenizer, model_info = lora_model, lora_tokenizer, lora_info
                    chip_bg = "linear-gradient(135deg, #7540ce, #9d67e8)"
                    border_color = "rgba(117, 64, 206, 0.4)"
                    shadow_color = "rgba(117, 64, 206, 0.3)"
                    model_details = "LoRA Fine-Tuned | 5 epochs | Context-Aware"
                elif mode == "Old Fine-Tuned" and old_info:
                    model, tokenizer, model_info = old_model, old_tokenizer, old_info
                    chip_bg = "linear-gradient(135deg, #e67e22, #f39c12)"
                    border_color = "rgba(230, 126, 34, 0.4)"
                    shadow_color = "rgba(230, 126, 34, 0.3)"
                    model_details = "Full Fine-Tuned | 3 epochs | 124M parameters"
                else:
                    model, tokenizer, model_info = base_model, base_tokenizer, base_info
                    chip_bg = "linear-gradient(135deg, #6c757d, #8a929a)"
                    border_color = "rgba(100, 100, 120, 0.3)"
                    shadow_color = "rgba(0, 0, 0, 0.3)"
                    model_details = "Base GPT-2 | 124M parameters"
                
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
                                padding: 3rem; box-shadow: 0 8px 32px {shadow_color};
                                margin: 1rem 0; animation: fadeInUp 0.6s ease;'>
                        <div style='text-align: center; margin-bottom: 2rem;'>
                            <div style='display: inline-block; background: {chip_bg};
                                        padding: 0.6rem 1.5rem; border-radius: 8px; font-size: 0.9rem;
                                        font-weight: 600; text-transform: uppercase; letter-spacing: 1px;
                                        box-shadow: 0 4px 15px {shadow_color}; color: white;'>
                                {model_info['name']}
                            </div>
                        </div>
                        <div style='font-family: "Poppins", sans-serif; font-size: 2rem;
                                    line-height: 1.5; color: #ffffff; font-weight: 500;
                                    text-align: center; padding: 1rem 0;'>
                            {headline}
                        </div>
                        <div style='margin-top: 2rem; padding-top: 1.5rem; border-top: 1px solid rgba(255, 255, 255, 0.1);
                                    text-align: center; font-size: 0.9rem; color: rgba(255, 255, 255, 0.6); font-family: "Space Grotesk", sans-serif;'>
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
        col1, col2, col3 = st.columns(3, gap="medium")
        
        with col1:
            st.markdown("""
            <div style='background: rgba(230, 126, 34, 0.1); border: 1px solid rgba(230, 126, 34, 0.3);
                        border-radius: 12px; padding: 1.5rem; text-align: center; height: 100%;'>
                <h4 style='color: #e67e22; margin: 0 0 1rem 0; font-size: 1.1rem;'>Old Fine-Tuned</h4>
                <p style='color: rgba(255, 255, 255, 0.8); margin: 0.5rem 0; font-size: 0.9rem; line-height: 1.6;'>
                    <strong>Architecture:</strong> GPT-2 Base<br>
                    <strong>Parameters:</strong> 124M (100% trained)<br>
                    <strong>Training:</strong> 8,000 headlines<br>
                    <strong>Epochs:</strong> 3<br>
                    <strong>Method:</strong> Full fine-tuning<br>
                    <strong>Issue:</strong> Catastrophic forgetting
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='background: rgba(117, 64, 206, 0.1); border: 1px solid rgba(117, 64, 206, 0.3);
                        border-radius: 12px; padding: 1.5rem; text-align: center; height: 100%;'>
                <h4 style='color: #7540ce; margin: 0 0 1rem 0; font-size: 1.1rem;'>LoRA Fine-Tuned ⭐</h4>
                <p style='color: rgba(255, 255, 255, 0.8); margin: 0.5rem 0; font-size: 0.9rem; line-height: 1.6;'>
                    <strong>Architecture:</strong> GPT-2 Base + LoRA<br>
                    <strong>Parameters:</strong> 124M (1.3% trained)<br>
                    <strong>Training:</strong> 8,000 headlines<br>
                    <strong>Epochs:</strong> 5<br>
                    <strong>Method:</strong> Parameter-efficient<br>
                    <strong>Benefit:</strong> Context-aware + humor
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style='background: rgba(100, 100, 120, 0.1); border: 1px solid rgba(100, 100, 120, 0.3);
                        border-radius: 12px; padding: 1.5rem; text-align: center; height: 100%;'>
                <h4 style='color: #8a929a; margin: 0 0 1rem 0; font-size: 1.1rem;'>Base GPT-2</h4>
                <p style='color: rgba(255, 255, 255, 0.8); margin: 0.5rem 0; font-size: 0.9rem; line-height: 1.6;'>
                    <strong>Architecture:</strong> GPT-2 Base<br>
                    <strong>Parameters:</strong> 124M<br>
                    <strong>Training:</strong> WebText corpus<br>
                    <strong>Epochs:</strong> Pre-trained<br>
                    <strong>Source:</strong> OpenAI/HuggingFace<br>
                    <strong>Use:</strong> General creative text
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
    
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