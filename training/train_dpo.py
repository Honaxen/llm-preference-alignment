"""
Fine-tunes an already instruction-tuned (SFT) model further using Direct
Preference Optimization (DPO) on the preference dataset built by
data/build_preference_dataset.py.

Why DPO instead of full RLHF (reward model + PPO):
DPO reframes the RLHF objective as a single classification-style loss
directly on (chosen, rejected) pairs -- no separate reward model to train,
no PPO rollouts, no instability from an RL loop. It's simpler to implement
correctly and has become the default starting point for preference
alignment in practice.

Uses LoRA (same approach as llm-fine-tuning) so this runs on the same
hardware -- DPO needs to keep a frozen reference copy of the policy model
in memory to compute its loss, which makes full fine-tuning even more
memory-hungry than SFT was; LoRA keeps that affordable.

Usage:
    python train_dpo.py \
        --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --sft_adapter ../../llm-fine-tuning/output/lora-adapter \
        --dataset ../data/preference_dataset.jsonl \
        --output_dir ../output/dpo-adapter
"""

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig


def load_model_and_tokenizer(base_model: str, sft_adapter: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float32 if device == "mps" else torch.float16,
    )

    if sft_adapter:
        # Start from the SFT checkpoint, not the raw base model -- DPO is
        # meant to refine an already instruction-tuned model, not replace
        # instruction-tuning entirely.
        print(f"Loading SFT adapter from {sft_adapter}")
        model = PeftModel.from_pretrained(model, sft_adapter, is_trainable=True)
        model = model.merge_and_unload()  # bake in SFT weights, then add a fresh DPO LoRA on top

    model = model.to(device)
    return model, tokenizer


def build_dpo_lora_config() -> LoraConfig:
    # Same rank/target-module choices as llm-fine-tuning's SFT LoRA config,
    # for a fair before/after comparison -- the only variable changing
    # between SFT and SFT+DPO should be the training objective, not the
    # adapter capacity.
    return LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )


def load_preference_dataset(path: str):
    dataset = load_dataset("json", data_files=path, split="train")
    print(f"Loaded {len(dataset)} preference pairs")
    return dataset


def main(args):
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, tokenizer = load_model_and_tokenizer(args.base_model, args.sft_adapter, device)
    dataset = load_preference_dataset(args.dataset)
    lora_config = build_dpo_lora_config()

    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        # beta controls how strongly DPO penalizes moving away from the
        # reference (pre-DPO) model's behavior -- higher beta = more
        # conservative updates, lower beta = trusts the preference labels more.
        beta=args.beta,
        logging_steps=10,
        save_strategy="epoch",
        bf16=False,  # kept off for MPS/CPU compatibility; enable on CUDA if available
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print("Starting DPO training...")
    trainer.train()

    print(f"Saving DPO adapter to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPO training on top of an SFT model")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--sft_adapter", default=None, help="Path to the SFT LoRA adapter from llm-fine-tuning")
    parser.add_argument("--dataset", required=True, help="Path to preference_dataset.jsonl")
    parser.add_argument("--output_dir", default="../output/dpo-adapter")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    args = parser.parse_args()

    main(args)
