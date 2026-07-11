"""
Builds a preference dataset (prompt, chosen, rejected) for DPO training.

For each seed prompt, two candidate responses are generated from the same
base model at different sampling temperatures -- a low-temperature response
(more focused) and a high-temperature response (more exploratory). An
LLM-as-judge then picks which one is actually better for that specific
prompt, since temperature alone isn't a reliable proxy for quality --
sometimes the "riskier" high-temperature response is more creative and wins.

Output format matches what TRL's DPOTrainer expects directly:
    {"prompt": "...", "chosen": "...", "rejected": "..."}

Usage:
    python build_preference_dataset.py \
        --model gemma3:12b \
        --judge_model gemma3:12b \
        --output ../data/preference_dataset.jsonl
"""

import argparse
import json
import re
import urllib.request
import urllib.error
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"

SEED_PROMPTS = [
    "Explain the difference between a list and a tuple in Python.",
    "Write a short email declining a meeting invitation politely.",
    "What are three tips for improving sleep quality?",
    "Summarize why the sky appears blue.",
    "Give me a recipe idea using chicken, rice, and broccoli.",
    "Explain what a REST API is to someone non-technical.",
    "Write a product description for a reusable water bottle.",
    "What's a good strategy for negotiating a salary?",
    "Explain the concept of compound interest with an example.",
    "Write a short, encouraging note for someone starting a new job.",
    "What are the main differences between supervised and unsupervised learning?",
    "Give advice for staying focused while working from home.",
    "Explain why regular exercise is important for mental health.",
    "Write a brief apology message for a delayed shipment.",
    "What's the difference between HTTP and HTTPS?",
]

JUDGE_PROMPT_TEMPLATE = """You are comparing two AI responses to the same prompt to decide which one is better.

Prompt:
{prompt}

Response A:
{response_a}

Response B:
{response_b}

Judge which response is more helpful, accurate, clear, and well-written. Reply with ONLY a JSON object, no other text:
{{"winner": "A" or "B", "reasoning": "one short sentence"}}
"""


def call_ollama(model: str, prompt: str, temperature: float, timeout: int = 60) -> str:
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
        return body.get("response", "").strip()


def judge_pair(judge_model: str, prompt: str, response_a: str, response_b: str) -> dict:
    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(prompt=prompt, response_a=response_a, response_b=response_b)
    raw_output = call_ollama(judge_model, judge_prompt, temperature=0.0)

    match = re.search(r"\{.*\}", raw_output, re.DOTALL)
    if not match:
        return {"winner": None, "reasoning": "judge did not return parseable JSON"}

    try:
        parsed = json.loads(match.group(0))
        winner = parsed.get("winner", "").upper()
        if winner not in ("A", "B"):
            winner = None
        return {"winner": winner, "reasoning": parsed.get("reasoning", "")}
    except json.JSONDecodeError:
        return {"winner": None, "reasoning": "judge JSON was malformed"}


def build_pair(model: str, judge_model: str, prompt: str) -> dict | None:
    response_a = call_ollama(model, prompt, temperature=0.3)   # focused
    response_b = call_ollama(model, prompt, temperature=1.1)   # exploratory

    verdict = judge_pair(judge_model, prompt, response_a, response_b)

    if verdict["winner"] == "A":
        chosen, rejected = response_a, response_b
    elif verdict["winner"] == "B":
        chosen, rejected = response_b, response_a
    else:
        return None  # skip pairs the judge couldn't decide on

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "judge_reasoning": verdict["reasoning"],
    }


def main(args):
    prompts = SEED_PROMPTS
    print(f"Building preference pairs for {len(prompts)} prompts...")

    pairs = []
    for i, prompt in enumerate(prompts, start=1):
        print(f"[{i}/{len(prompts)}] {prompt[:60]}...")
        try:
            pair = build_pair(args.model, args.judge_model, prompt)
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            print(f"    generation failed: {e}")
            continue

        if pair is None:
            print("    judge could not decide, skipped")
            continue

        pairs.append(pair)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for pair in pairs:
            # DPOTrainer only needs prompt/chosen/rejected -- keep judge_reasoning
            # out of the training file itself, save it separately for inspection.
            f.write(json.dumps({
                "prompt": pair["prompt"],
                "chosen": pair["chosen"],
                "rejected": pair["rejected"],
            }) + "\n")

    reasoning_path = output_path.with_suffix(".reasoning.json")
    with open(reasoning_path, "w") as f:
        json.dump(pairs, f, indent=2)

    print(f"\nBuilt {len(pairs)}/{len(prompts)} preference pairs.")
    print(f"Training data: {output_path}")
    print(f"Judge reasoning (for inspection): {reasoning_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a DPO preference dataset via self-play + LLM judging")
    parser.add_argument("--model", default="gemma3:12b", help="Model to generate candidate responses")
    parser.add_argument("--judge_model", default="gemma3:12b", help="Model to judge which response is better")
    parser.add_argument("--output", default="../data/preference_dataset.jsonl")
    args = parser.parse_args()

    main(args)
