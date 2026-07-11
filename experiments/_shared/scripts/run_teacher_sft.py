#!/usr/bin/env python3
"""SFT student model on teacher-generated responses using HuggingFace Trainer."""
import os, sys, json, argparse
sys.path.insert(0, "/path/to/EasyOPD")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_model", default="/path/to/models/phi4-mini-sft-warmup-10k-qwen-lr2e-6/checkpoint-40")
    parser.add_argument("--data_path", default="/path/to/EasyOPD/experiments/benchmark/teacher_sft_data/teacher_sft_train.jsonl")
    parser.add_argument("--output_dir", default="/path/to/EasyOPD/experiments/benchmark/checkpoints/teacher_sft_phi4mini")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=2048)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
    from datasets import Dataset
    import torch

    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data
    data = []
    with open(args.data_path) as f:
        for line in f:
            data.append(json.loads(line))
    print(f"Loaded {len(data)} training samples")

    # Tokenize using student's chat template
    def tokenize_fn(examples):
        texts = []
        for msgs in examples["messages"]:
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        encodings = tokenizer(texts, truncation=True, max_length=args.max_length, padding="max_length")
        encodings["labels"] = encodings["input_ids"].copy()
        return encodings

    dataset = Dataset.from_list([{"messages": item["messages"]} for item in data])
    tokenized = dataset.map(tokenize_fn, batched=True, batch_size=100, remove_columns=["messages"])
    print(f"Tokenized dataset: {len(tokenized)} samples")

    model = AutoModelForCausalLM.from_pretrained(
        args.student_model, trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    model.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        dataloader_num_workers=4,
        report_to="none",
        gradient_checkpointing=True,
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "backward_prefetch": "backward_pre",
            "forward_prefetch": False,
            "use_orig_params": True,
        },
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized, processing_class=tokenizer)
    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "final"))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))
        print(f"Model saved to {os.path.join(args.output_dir, 'final')}")

if __name__ == "__main__":
    main()
