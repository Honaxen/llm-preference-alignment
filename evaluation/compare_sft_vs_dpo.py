"""
Head-to-head comparison between the SFT-only model (from llm-fine-tuning)
and the SFT+DPO model (from training/train_dpo.py), on a held-out set of
prompts neither model was trained on.

This is the script that actually answers the project's core question:
did DPO measurably change behavior, or just add training time? A single
eyeballed example isn't evidence -- a win rate across many prompts is.

To avoid position bias (judges tend to slightly favor whichever response
is shown first), each pair is judged twice with the order swapped, and a
model only counts as the winner if it wins both orderings. Anything else
is scored a tie.

Usage:
    python compare_sft_vs_dpo.py \
        --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --sft_adapter ../../llm-fine-tuning/output/lora-adapter \
        --dpo_adapter ../output/dpo-adapter \
        --judge_model gemma3:12b \
        --output ../evaluation/results/head_to_head.json
"""

import argparse
import json
import re
import urllib.request
import urllib.error
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

OLLAMA_URL = "http://localhost:11434/api/generate"

# Held-out prompts -- deliberately different from data/build_preference_dataset.py's
# seed prompts, so this evaluates generalization, not memorized preference pairs.
HELD_OUT_PROMPTS = [
    "How do I politely ask a coworker to stop interrupting me in meetings?",
    "Explain what caching means in software, using a real-world analogy.",
    "Suggest a simple weekly meal plan for someone who's busy on weeknights.",
    "What should I consider before signing a one-year apartment lease?",
    "Write a short congratulatory message for a friend's promotion.",
    "Explain the pros and cons of remote work versus office work.",
    "What's a good way to start learning a new programming language?",
    "Give three tips for giving constructive feedback to a teammate.",
]

JUDGE_PROMPT_TEMPLATE = """You are comparing two AI responses to the same prompt to decide which one is better.

Prompt:
{prompt}

Response 1:
{response_1}

Response 2:
{response_2}

Judge which response is more helpful, natural, and well-aligned with what a good assistant would say. Reply with ONLY a JSON object, no other text:
{{"winner": "1" or "2", "reasoning": "one short sentence"}}
"""


def load_model(base_model: str, adapter_path: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float32 if device == "mps" else torch.float16,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, device: str, max_new_tokens: int = 200) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    generated = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def call_judge(judge_model: str, prompt: str, response_1: str, response_2: str) -> str | None:
    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(prompt=prompt, response_1=response_1, response_2=response_2)

    payload = json.dumps({
        "model": judge_model,
        "prompt": judge_prompt,
        "stream": False,
        "options": {"temperature": 0.0},
    }).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
            raw_output = body.get("response", "").strip()
    except (urllib.error.URLError, urllib.error.HTTPError):
        return None

    match = re.search(r"\{.*\}", raw_output, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        winner = str(parsed.get("winner", "")).strip()
        return winner if winner in ("1", "2") else None
    except json.JSONDecodeError:
        return None


def judge_pair_both_orders(judge_model: str, prompt: str, sft_response: str, dpo_response: str) -> str:
    """
    Returns "dpo", "sft", or "tie".
    Judges twice with order swapped to cancel out position bias --
    a model only wins if it wins under both orderings.
    """
    # Order A: SFT first, DPO second
    winner_a = call_judge(judge_model, prompt, sft_response, dpo_response)
    # Order B: DPO first, SFT second
    winner_b = call_judge(judge_model, prompt, dpo_response, sft_response)

    dpo_won_a = winner_a == "2"
    sft_won_a = winner_a == "1"
    dpo_won_b = winner_b == "1"
    sft_won_b = winner_b == "2"

    if dpo_won_a and dpo_won_b:
        return "dpo"
    if sft_won_a and sft_won_b:
        return "sft"
    return "tie"


def main(args):
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading SFT-only model...")
    sft_model, tokenizer = load_model(args.base_model, args.sft_adapter, device)

    print("Loading SFT+DPO model...")
    dpo_model, _ = load_model(args.base_model, args.dpo_adapter, device)

    results = []
    for i, prompt in enumerate(HELD_OUT_PROMPTS, start=1):
        print(f"[{i}/{len(HELD_OUT_PROMPTS)}] {prompt[:60]}...")

        sft_response = generate_response(sft_model, tokenizer, prompt, device)
        dpo_response = generate_response(dpo_model, tokenizer, prompt, device)

        winner = judge_pair_both_orders(args.judge_model, prompt, sft_response, dpo_response)

        results.append({
            "prompt": prompt,
            "sft_response": sft_response,
            "dpo_response": dpo_response,
            "winner": winner,
        })
        print(f"    winner: {winner}")

    dpo_wins = sum(1 for r in results if r["winner"] == "dpo")
    sft_wins = sum(1 for r in results if r["winner"] == "sft")
    ties = sum(1 for r in results if r["winner"] == "tie")
    total = len(results)

    summary = {
        "total_prompts": total,
        "dpo_wins": dpo_wins,
        "sft_wins": sft_wins,
        "ties": ties,
        "dpo_win_rate_pct": round(dpo_wins / total * 100, 1),
        "sft_win_rate_pct": round(sft_wins / total * 100, 1),
        "tie_rate_pct": round(ties / total * 100, 1),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print("\n=== SFT vs SFT+DPO Head-to-Head ===")
    print(f"DPO win rate: {summary['dpo_win_rate_pct']}%")
    print(f"SFT win rate: {summary['sft_win_rate_pct']}%")
    print(f"Tie rate:     {summary['tie_rate_pct']}%")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Head-to-head comparison: SFT-only vs SFT+DPO")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--sft_adapter", required=True, help="Path to the SFT-only LoRA adapter")
    parser.add_argument("--dpo_adapter", required=True, help="Path to the SFT+DPO LoRA adapter")
    parser.add_argument("--judge_model", default="gemma3:12b")
    parser.add_argument("--output", default="../evaluation/results/head_to_head.json")
    args = parser.parse_args()

    main(args)
