"""
Fake News Headline Generator - Streamlit Web UI
Generates humorous/exaggerated fake news headlines using Fine-Tuned GPT-2 Large
Fine-tuned on real fake news dataset for maximum authenticity and humor
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
    Load Fine-Tuned GPT-2 Large model and tokenizer
    Uses fine-tuned model if available, otherwise falls back to base model
    Returns: model, tokenizer, and is_finetuned flag
    """
    import os
    
    fine_tuned_path = "./fine_tuned_model/final"
    base_model_name = "gpt2-large"
    
    # Check if fine-tuned model exists
    if os.path.exists(fine_tuned_path):
        st.info("🎯 Using fine-tuned model trained on fake news dataset!")
        model_path = fine_tuned_path
        is_finetuned = True
    else:
        st.warning("⚠️ Fine-tuned model not found. Using base GPT-2 Large. Run 'python train_model.py' to train the model.")
        model_path = base_model_name
        is_finetuned = False
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Set padding token (GPT-2 doesn't have one by default)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer, is_finetuned

def generate_headline(topic, model, tokenizer, is_finetuned=False):
    """
    Generate a fake news headline based on the given topic
    
    Args:
        topic: User-provided topic string
        model: Fine-tuned GPT-2 Large model
        tokenizer: GPT-2 tokenizer
        is_finetuned: Whether using fine-tuned model
    
    Returns:
        Generated headline string
    """
    # Use simpler prompt for fine-tuned model as it's already trained on fake news
    if is_finetuned:
        prompt = f"Generate a fake news headline: about {topic}"
    else:
        # More detailed prompt for base model
        prompt = f"""You are a satirical fake news headline generator. Create one ridiculous, absurd, and hilarious fake news headline about: {topic}

The headline should be:
- Completely outrageous and funny
- Unexpected and bizarre
- Written in news headline style
- Creative and imaginative

Headline:"""
    
    # Tokenize the input
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    # Adjust parameters based on model type
    temp = 1.3 if is_finetuned else 1.5
    max_new_tokens = 50 if is_finetuned else 40
    
    # Generate headline with parameters optimized for humor
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=inputs.shape[1] + max_new_tokens,
            do_sample=True,
            temperature=temp,
            top_p=0.95,
            top_k=100,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3
        )
    
    # Decode the generated text
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the headline part
    if is_finetuned:
        # For fine-tuned model, remove the prompt
        headline = full_output.replace(prompt, "").strip()
    else:
        # For base model, extract after "Headline:"
        if "Headline:" in full_output:
            headline = full_output.split("Headline:")[-1].strip()
        else:
            headline = full_output
    
    # Clean up the headline - take only the first complete sentence or line
    headline = headline.split('\n')[0].strip()
    
    # Remove any leading colons or dashes
    headline = headline.lstrip(':- ')
    
    # Ensure it ends properly
    if headline and not headline.endswith(('.', '!', '?')):
        if '.' in headline:
            headline = headline.split('.')[0] + '.'
        elif '!' in headline:
            headline = headline.split('!')[0] + '!'
        else:
            headline = headline.rstrip() + '!'
    
    return headline if headline else "Breaking: Something Absolutely Ridiculous Just Happened!"

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
        model, tokenizer, is_finetuned = load_model()
    
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
                    headline = generate_headline(topic, model, tokenizer, is_finetuned)
                    
                    # Display the result in a nice box
                    st.markdown("### Generated Headline:")
                    st.success(headline)
                    
                    # Add a fun disclaimer
                    st.caption("⚠️ This is AI-generated fiction for entertainment purposes only!")
            else:
                st.warning("Please enter a topic first!")
    
    # Footer
    st.markdown("---")
    model_status = "Fine-Tuned" if is_finetuned else "Base"
    st.markdown(
        f"<div style='text-align: center; color: gray;'>"
        f"Powered by {model_status} GPT-2 Large (774M parameters) from HuggingFace 🤗"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
