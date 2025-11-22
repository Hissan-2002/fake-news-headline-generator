"""
Fake News Headline Generator - Streamlit Web UI
Generates humorous/exaggerated fake news headlines using GPT-2 Large
"""

import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Set page configuration
st.set_page_config(
    page_title="Fake News Headline Generator",
    page_icon="📰",
    layout="centered"
)

# Cache the model and tokenizer to load only once
@st.cache_resource
def load_model():
    """
    Load GPT-2 Large model and tokenizer
    Using gpt2-large for significantly better quality and coherence
    Returns: model and tokenizer
    """
    model_name = "gpt2-large"  # 774M parameters - much more advanced than distilgpt2
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Set padding token (GPT-2 doesn't have one by default)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def generate_headline(topic, model, tokenizer):
    """
    Generate a fake news headline based on the given topic
    
    Args:
        topic: User-provided topic string
        model: GPT-2 Large model
        tokenizer: GPT-2 tokenizer
    
    Returns:
        Generated headline string
    """
    # Create a system-style prompt that instructs the model on its behavior
    prompt = f"""You are a satirical fake news headline generator. Create one ridiculous, absurd, and hilarious fake news headline about: {topic}

The headline should be:
- Completely outrageous and funny
- Unexpected and bizarre
- Written in news headline style
- Creative and imaginative

Headline:"""
    
    # Tokenize the input
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate headline with parameters maximized for absurdity and humor
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=inputs.shape[1] + 40,  # Allow 40 tokens for the headline
            do_sample=True,          # Enable sampling for variety
            temperature=1.5,         # VERY high temperature for maximum absurdity
            top_p=0.98,              # Very high top_p for wild vocabulary
            top_k=100,               # Increased top_k for more options
            num_return_sequences=1,  # Generate one headline
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.3,  # Stronger penalty to avoid boring repetition
            no_repeat_ngram_size=3   # Prevent repetitive phrases
        )
    
    # Decode the generated text
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the headline part (after "Headline:")
    if "Headline:" in full_output:
        headline = full_output.split("Headline:")[-1].strip()
    else:
        headline = full_output
    
    # Clean up the headline - take only the first complete sentence or line
    headline = headline.split('\n')[0].strip()
    
    # Ensure it ends properly
    if not headline.endswith(('.', '!', '?')):
        if '.' in headline:
            headline = headline.split('.')[0] + '.'
        elif '!' in headline:
            headline = headline.split('!')[0] + '!'
        else:
            headline = headline + '!'
    
    return headline

# Main UI
def main():
    """
    Main function to render the Streamlit UI
    """
    # Title and description
    st.title("📰 Fake News Headline Generator")
    st.markdown("Generate humorous and exaggerated fake news headlines using AI!")
    st.markdown("---")
    
    # Load model (cached, so it only loads once)
    with st.spinner("Loading AI model... (this may take a moment on first run)"):
        model, tokenizer = load_model()
    
    # Center the input elements
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        # Topic input
        topic = st.text_input(
            "Enter a topic:",
            placeholder="e.g., cats, politics, technology...",
            help="Enter any topic and we'll generate a fake news headline!"
        )
        
        # Generate button
        generate_button = st.button("🚀 Generate Headline", use_container_width=True)
        
        # Display area for generated headline
        if generate_button:
            if topic.strip():
                with st.spinner("Generating your fake news headline..."):
                    # Generate the headline
                    headline = generate_headline(topic, model, tokenizer)
                    
                    # Display the result in a nice box
                    st.markdown("### Generated Headline:")
                    st.success(headline)
                    
                    # Add a fun disclaimer
                    st.caption("⚠️ This is AI-generated fiction for entertainment purposes only!")
            else:
                st.warning("Please enter a topic first!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Powered by GPT-2 Large (774M parameters) from HuggingFace 🤗"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
