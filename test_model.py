"""
Test the fine-tuned model
Quick script to test the fine-tuned model's headline generation
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_test_headlines(model_path="./fine_tuned_model/final"):
    """
    Test the fine-tuned model with sample topics
    """
    print("Loading fine-tuned model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Test topics
    test_topics = [
        "aliens",
        "pizza",
        "politics",
        "cats",
        "technology",
        "climate change",
        "celebrities",
        "sports"
    ]
    
    print("\n" + "="*60)
    print("Testing Fine-Tuned Model")
    print("="*60 + "\n")
    
    for topic in test_topics:
        prompt = f"Generate a fake news headline: about {topic}"
        
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + 40,
                do_sample=True,
                temperature=1.3,
                top_p=0.95,
                top_k=100,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        headline = generated.replace(prompt, "").strip()
        
        # Extract first sentence
        if '\n' in headline:
            headline = headline.split('\n')[0]
        
        print(f"Topic: {topic}")
        print(f"Headline: {headline}")
        print("-" * 60 + "\n")

if __name__ == "__main__":
    generate_test_headlines()
