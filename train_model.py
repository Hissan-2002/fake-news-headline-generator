"""
Fine-tune GPT-2 Large on Fake News Dataset
This script prepares the dataset and fine-tunes the model for headline generation
"""

import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import os

# Configuration
MODEL_NAME = "gpt2-large"
OUTPUT_DIR = "./fine_tuned_model"
DATASET_FILE = "Fake.csv"
MAX_LENGTH = 128
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 2e-5

def load_and_prepare_data(csv_file, sample_size=None):
    """
    Load the fake news CSV and prepare it for training
    
    Args:
        csv_file: Path to the CSV file
        sample_size: Optional number of samples to use (for testing)
    
    Returns:
        Prepared dataset
    """
    print(f"Loading dataset from {csv_file}...")
    
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Use only the 'title' column for headline generation
    # Filter out any null titles
    df = df[['title']].dropna()
    
    # Sample if specified (useful for quick testing)
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    print(f"Loaded {len(df)} headlines")
    
    # Create training prompts
    # Format: "Generate a fake news headline: [TITLE]"
    df['text'] = df['title'].apply(lambda x: f"Generate a fake news headline: {x}")
    
    # Convert to HuggingFace dataset
    dataset = Dataset.from_pandas(df[['text']])
    
    return dataset

def tokenize_function(examples, tokenizer):
    """
    Tokenize the text data
    """
    # Tokenize with padding and truncation
    tokenized = tokenizer(
        examples['text'],
        truncation=True,
        max_length=MAX_LENGTH,
        padding='max_length',
        return_tensors='pt'
    )
    
    # For causal language modeling, labels are the same as input_ids
    tokenized['labels'] = tokenized['input_ids'].clone()
    
    return tokenized

def main():
    """
    Main training function
    """
    print("=" * 60)
    print("Fine-tuning GPT-2 Large on Fake News Headlines")
    print("=" * 60)
    
    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    # Load tokenizer and model
    print(f"\nLoading {MODEL_NAME} model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    
    # Set padding token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    
    # Load and prepare dataset
    dataset = load_and_prepare_data(DATASET_FILE)
    
    # Tokenize dataset
    print("\nTokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Split into train and validation
    print("\nSplitting dataset...")
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're doing causal language modeling, not masked
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        warmup_steps=100,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        push_to_hub=False,
        report_to="none"  # Disable wandb/tensorboard unless you want it
    )
    
    # Initialize Trainer
    print("\nInitializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # Train the model
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    print(f"\nThis will take a while. Grab a coffee! ☕")
    print(f"Training for {EPOCHS} epochs with batch size {BATCH_SIZE}\n")
    
    trainer.train()
    
    # Save the final model
    print("\n" + "=" * 60)
    print("Training complete! Saving model...")
    print("=" * 60)
    
    final_model_path = "./fine_tuned_model/final"
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    print(f"\n✅ Model saved to: {final_model_path}")
    print("\nYou can now use this fine-tuned model in web_ui.py!")
    print("\nTo test the model, run: python test_model.py")

if __name__ == "__main__":
    main()
