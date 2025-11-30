import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from datasets import Dataset
import os
import re

# Configuration 
MODEL_NAME = "gpt2" 
OUTPUT_DIR = "./fine_tuned_model_lora"
DATASET_FILE = "Fake.csv"
MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 5  
LEARNING_RATE = 2e-4  
SAMPLE_SIZE = 8000

# Preprocessing
def extract_keywords_from_title(title):
  
    cleaned = re.sub(r'(breaking|exclusive|shocking|revealed|exposed|just|now)[:;\s]*', '', title, flags=re.IGNORECASE)
    cleaned = re.sub(r'[!?]{2,}', '', cleaned)
    topics = [
        # Political
        'Trump', 'Donald Trump', 'Obama', 'Barack Obama', 'Clinton', 'Hillary Clinton',
        'Biden', 'White House', 'Congress', 'Senate', 'Democrats', 'Republicans',
        'GOP', 'FBI', 'CIA', 'Russia', 'Putin', 'Election', 'Investigation',
        # General
        'Police', 'Court', 'Judge', 'Lawyer', 'Doctor', 'Hospital', 'School',
        'Teacher', 'Student', 'Professor', 'Scientist', 'Study', 'Research',
        # Tech/Business
        'Facebook', 'Twitter', 'Google', 'Apple', 'Tesla', 'Amazon',
        'iPhone', 'Android', 'Bitcoin', 'Cryptocurrency', 'AI', 'Robot',
        # Other
        'Celebrity', 'Hollywood', 'Netflix', 'Video Game', 'Movie', 'TV'
    ]
    
    # Check for known topics (prioritize longer matches)
    for keyword in sorted(topics, key=len, reverse=True):
        if keyword.lower() in title.lower():
            return keyword
    
    # Extract first 2-4 meaningful words as topic
    words = [w for w in cleaned.strip().split() if len(w) > 2][:4]
    topic = ' '.join(words) if words else title.split()[:3]
    
    return ' '.join(topic).strip('.,!?:') if isinstance(topic, list) else topic.strip('.,!?:')

def load_and_prepare_data(csv_file, sample_size=None):
    print(f"Loading dataset from {csv_file}...")
    
    df = pd.read_csv(csv_file)
    df = df[['title']].dropna()
    df = df[df['title'].str.len() > 30]
    df = df[df['title'].str.len() < 250]
    df = df.drop_duplicates(subset=['title'])
    
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    print(f"Loaded {len(df)} quality headlines")
    formatted_data = []
    
    for _, row in df.iterrows():
        title = row['title'].strip()
        topic = extract_keywords_from_title(title)
        
        prompts = [
            f"Generate a fake news headline about {topic}: {title}<|endoftext|>",
            f"Topic: {topic}\nHeadline: {title}<|endoftext|>",
            f"Create a satirical headline about {topic}: {title}<|endoftext|>",
            f"Fake news: {topic}\n{title}<|endoftext|>",
            f"{topic}: {title}<|endoftext|>",
        ]
        formatted_data.append({'text': prompts[hash(title) % len(prompts)]})
    
    print(f"Created {len(formatted_data)} training examples with context")
    
    dataset = Dataset.from_list(formatted_data)
    return dataset

def tokenize_function(examples, tokenizer):
    result = tokenizer(
        examples['text'],
        truncation=True,
        max_length=MAX_LENGTH,
        padding='max_length',
        return_tensors=None
    )
    result['labels'] = result['input_ids'].copy()
    return result

def main():
    print("\n" + "=" * 80)
    print("\nConfiguration:")
    print(f"   • Model: {MODEL_NAME}")
    print(f"   • Method: LoRA (Parameter-Efficient Fine-Tuning)")
    print(f"   • Training samples: {SAMPLE_SIZE}")
    print(f"   • Epochs: {EPOCHS}")
    print(f"   • Batch size: {BATCH_SIZE}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device.upper()}")
    print(f"Estimated time: {'20-30 min' if device == 'cuda' else '45-60 min'}")
    print(f"\nLoading {MODEL_NAME} model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    print("\nConfiguring LoRA (Low-Rank Adaptation)...")
    lora_config = LoraConfig(
        r=16,  
        lora_alpha=32,  
        target_modules=["c_attn", "c_proj", "c_fc"],  
        lora_dropout=0.1, 
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"LoRA applied successfully!")
    print(f"   • Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"   • Total parameters: {total_params:,}")
    print(f"   • Memory efficient: {100 - (100 * trainable_params / total_params):.1f}% parameters frozen")
    
    # Load dataset
    print("\n" + "-" * 80)
    dataset = load_and_prepare_data(DATASET_FILE, sample_size=SAMPLE_SIZE)
    
    # Tokenize
    print("\nTokenizing with context awareness...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    # Split
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    
    print(f"\nDataset split:")
    print(f"   • Training: {len(train_dataset)} samples")
    print(f"   • Validation: {len(eval_dataset)} samples")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training arguments optimized for best accuracy
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        
        # LoRA-optimized settings
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        eval_strategy="steps",
        eval_steps=200,  
        save_steps=400,  
        save_total_limit=3,
        load_best_model_at_end=True,
        
        # Learning - optimized for convergence
        learning_rate=LEARNING_RATE,
        warmup_steps=200, 
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,  
        
        # Optimization
        optim="adamw_torch",
        lr_scheduler_type="cosine",  
        
        # Logging
        logging_dir='./logs',
        logging_steps=50,
        logging_first_step=True,
        
        # Performance
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        
        # Output
        report_to="none",
        disable_tqdm=False,

        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    # Trainer
    print("\n" + "-" * 80)
    print("Initializing LoRA trainer...\n")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    try:
        train_result = trainer.train()
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE!")
        print("=" * 80)
        print(f"\nFinal Training Loss: {train_result.training_loss:.4f}")
        
        # Save model
        print("\nSaving LoRA model...")
        final_model_path = OUTPUT_DIR + "/final"
        
        # Save LoRA adapters
        model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        
        # Save merged model for easy deployment
        print("Merging LoRA weights with base model...")
        merged_model_path = OUTPUT_DIR + "/final_merged"
        os.makedirs(merged_model_path, exist_ok=True)
        
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged_model_path)
        tokenizer.save_pretrained(merged_model_path)
        trainer.save_model(OUTPUT_DIR + "/interrupted")
        
    except Exception as e:
        print(f"\n\nTraining failed: {str(e)}")
        print("\nIf you get 'peft not found', install it:")
        print("   pip install peft")
        raise

if __name__ == "__main__":
    main()
