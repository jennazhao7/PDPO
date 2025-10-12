# train_dpo_stage1.py

from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_name = "openai-community/gpt2-large"
    dataset_path = "./dpo_train_ready.jsonl"
    output_dir = "./models/M1"
    lr = 5e-5
    batch_size = 4
    epochs = 3

    # Load dataset
    dataset = load_dataset("json", data_files=dataset_path)["train"]

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Configure DPO (aligned with PROPS-2025)
    config = DPOConfig(
        model_name_or_path=model_name,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,  # PROPS-2025 standard
        gradient_checkpointing=True,    # Memory optimization
        max_length=512,                 # PROPS-2025 standard
        num_train_epochs=epochs,
        lr_scheduler_type="cosine",     # PROPS-2025 standard
        warmup_steps=10,                # PROPS-2025 standard
        optim="paged_adamw_32bit",      # PROPS-2025 standard
        bf16=True,                      # PROPS-2025 standard (memory efficient)
        output_dir=output_dir,
        save_strategy="no",             # PROPS-2025 standard
        logging_steps=1,                # PROPS-2025 standard
        report_to=None                  # Disable wandb
    )

    trainer = DPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=config,
        train_dataset=dataset
    )

    trainer.train()
    trainer.save_model(output_dir)
    print(f"✅ Stage-1 model saved to {output_dir}")

if __name__ == "__main__":
    main()
