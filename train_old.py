import pandas as pd
import torch
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import os


MODEL_NAME = "gpt2"  
OUTPUT_DIR = "./fine_tuned_model"
DATASET_FILE = "Fake.csv"
MAX_LENGTH = 80  
BATCH_SIZE = 4  
EPOCHS = 3  
LEARNING_RATE = 5e-5  
SAMPLE_SIZE = 8000  

def load_and_prepare_data(csv_file, sample_size=None):
    print(f"Loading dataset from {csv_file}...")
    
    df = pd.read_csv(csv_file)
    df = df[['title']].dropna()
    df = df[df['title'].str.len() > 20]  
    df = df[df['title'].str.len() < 200]  
    
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    print(f"Loaded {len(df)} quality headlines")
    df['text'] = df['title'].apply(lambda x: f"Fake News: {x}<|endoftext|>")
    
    dataset = Dataset.from_pandas(df[['text']])
    return dataset

def tokenize_function(examples, tokenizer):
    result = tokenizer(
        examples['text'],
        truncation=True,
        max_length=MAX_LENGTH,
        padding='max_length',
    )
    result['labels'] = result['input_ids'].copy()
    return result

def main():
    print("\n" + "=" * 70)
    print("OPTIMIZED FAKE NEWS HEADLINE GENERATOR - FINE-TUNING")
    print("=" * 70)
    print("\n⚡ Configuration:")
    print(f"   • Model: {MODEL_NAME} (124M parameters - fast & efficient)")
    print(f"   • Training samples: {SAMPLE_SIZE}")
    print(f"   • Epochs: {EPOCHS}")
    print(f"   • Batch size: {BATCH_SIZE}")
    print(f"   • Max length: {MAX_LENGTH}")
    
    # Device info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device.upper()}")
    
    if device == "cpu":
        print("Estimated time: 60-90 minutes")
    else:
        print("Estimated time: 15-20 minutes")
    
    # Load model and tokenizer
    print(f"\n📥 Loading {MODEL_NAME} model...")
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
        model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    except Exception as e:
        print(f"\nError loading model: {str(e)}")
        try:
            tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
            model = GPT2LMHeadModel.from_pretrained(MODEL_NAME, local_files_only=True)
            print("Loaded from cache!")
        except:
            print("\nModel not found in cache either.")
            raise
    
    # Set padding token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    
    # Load and prepare dataset
    print("\n" + "-" * 70)
    dataset = load_and_prepare_data(DATASET_FILE, sample_size=SAMPLE_SIZE)
    
    # Tokenize
    print("\nTokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    # Split dataset
    print("\nSplitting into train/validation...")
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    
    print(f"• Training: {len(train_dataset)} samples")
    print(f"• Validation: {len(eval_dataset)} samples")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Optimized training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        
        # Optimization for speed
        gradient_accumulation_steps=2,  
        gradient_checkpointing=False,  
        
        # Evaluation and saving
        eval_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        save_total_limit=2,
        load_best_model_at_end=True,
        
        # Learning parameters
        learning_rate=LEARNING_RATE,
        warmup_steps=100,
        weight_decay=0.01,
        max_grad_norm=1.0,
        
        # Logging
        logging_dir='./logs',
        logging_steps=50,
        logging_first_step=True,
        
        # Performance
        dataloader_num_workers=0,  
        fp16=torch.cuda.is_available(),
        
        # Output
        report_to="none",
        disable_tqdm=False,
        push_to_hub=False,
        
        # Best model selection
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    # Initialize trainer
    print("\n" + "-" * 70)
    print("🎯 Initializing trainer...\n")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # Calculate total steps
    total_steps = (len(train_dataset) // (BATCH_SIZE * 2)) * EPOCHS
    print(f"Total training steps: {total_steps}")
    print(f"Model will be saved every 1000 steps")
    print(f"Evaluation every 500 steps\n")
    
    # Train
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    print("\nGrab a coffee and relax! Training will complete automatically.")
    print("Progress bar will show below:\n")
    
    try:
        # Train the model
        train_result = trainer.train()
        
        # Training complete
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE!")
        print("=" * 70)
        
        # Print training metrics
        print(f"\nFinal Training Loss: {train_result.training_loss:.4f}")
        
        # Save final model
        print("\nSaving fine-tuned model...")
        final_model_path = OUTPUT_DIR + "/final"
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        
        print(f"\nModel successfully saved to: {final_model_path}")
        print("\n" + "=" * 70)
        print("FINE-TUNING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nNext Steps:")
        print("   1. Test the model:    python test_model.py")
        print("   2. Run the web UI:    python -m streamlit run web_ui.py")
        print("   3. Generate headlines and have fun!\n")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print("Saving current progress...")
        trainer.save_model(OUTPUT_DIR + "/interrupted")
        print("Progress saved. You can resume or use this checkpoint.\n")
        
    except Exception as e:
        print(f"\n\nTraining failed with error:")
        print(f"{str(e)}\n")
        raise
if __name__ == "__main__":
    main()