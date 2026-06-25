"""Llama 3 QLoRA Colab recipe.

This file is intentionally small. It is a starter script, not a benchmark run.
"""

from __future__ import annotations

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from peft_doctor import diagnose_peft, recommend_target_modules


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--data", default="sample_data.jsonl")
    parser.add_argument("--output-dir", default="outputs/llama3_qlora_colab")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset = load_dataset("json", data_files=args.data, split="train")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=recommend_target_modules(model_name=args.model),
    )
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-4,
        max_seq_length=2048,
        max_steps=args.max_steps,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
    )
    report = diagnose_peft(
        model=model,
        tokenizer=tokenizer,
        peft_config=peft_config,
        training_args=training_args,
        train_dataset=dataset,
        sequence_length=2048,
        model_name=args.model,
    )
    print(report.to_markdown())
    if report.has_errors or args.dry_run:
        return

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
