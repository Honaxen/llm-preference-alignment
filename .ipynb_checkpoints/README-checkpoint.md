# LLM Preference Alignment (DPO)

Aligning an instruction-tuned LLM to human preferences using Direct Preference Optimization (DPO) — and measuring exactly how much that changes behavior compared to instruction-tuning alone.

---

## What This Project Demonstrates

`llm-fine-tuning` showed how to teach a model to follow instructions (SFT).
This one shows how to teach a model which of two correct-ish answers is actually better — the step that turns an instruction-follower into something that behaves like a real assistant.

| Concern | Solution |
|---|---|
| How do you teach "better", not just "correct"? | Preference dataset: prompt + chosen response + rejected response |
| How do you train on preferences directly? | Direct Preference Optimization (DPO), no separate reward model or PPO loop |
| Did alignment actually change behavior? | Before/after comparison: SFT-only vs SFT+DPO, judged head-to-head |
| Is the improvement real or just noise? | Win rate on held-out prompts, with position-bias-controlled judging |

---

## Architecture

```
Preference Dataset  →  self-play (two temperatures) + LLM-as-judge picks chosen/rejected
  ↓
DPO Training  →  LoRA on top of the SFT adapter, trained on preference pairs
  ↓
Head-to-Head Evaluation  →  SFT-only vs SFT+DPO, judged on held-out prompts
  ↓
Win Rate Report
```

---

## Project Structure

```
llm-preference-alignment/
├── data/
│   └── build_preference_dataset.py   — self-play + LLM-as-judge preference pairs
├── training/
│   └── train_dpo.py                  — DPO training on top of the SFT LoRA adapter
├── evaluation/
│   ├── compare_sft_vs_dpo.py         — head-to-head, position-bias-controlled judging
│   └── results/
├── tests/
│   └── test_alignment.py             — 7/7 passing
├── docs/
│   └── architecture.md
└── requirements.txt
```

---

## Getting Started

```bash
pip install -r requirements.txt
ollama serve
ollama pull gemma3:12b
```

### 1. Build the preference dataset

```bash
python data/build_preference_dataset.py \
  --model gemma3:12b \
  --judge_model gemma3:12b \
  --output data/preference_dataset.jsonl
```

### 2. Run DPO training on top of the SFT adapter

```bash
python training/train_dpo.py \
  --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --sft_adapter ../llm-fine-tuning/output/lora-adapter \
  --dataset data/preference_dataset.jsonl \
  --output_dir output/dpo-adapter
```

### 3. Compare SFT-only vs SFT+DPO head-to-head

```bash
python evaluation/compare_sft_vs_dpo.py \
  --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --sft_adapter ../llm-fine-tuning/output/lora-adapter \
  --dpo_adapter output/dpo-adapter \
  --judge_model gemma3:12b
```

Example output *(illustrative — replace with your own run)*:
```
=== SFT vs SFT+DPO Head-to-Head ===
DPO win rate: 62.5%
SFT win rate: 12.5%
Tie rate:     25.0%
```

### 4. Run tests

```bash
pytest tests/ -v
```

---

## Stack

Python · PyTorch · PEFT · TRL · HuggingFace · Ollama

---

## What I Learned

**A preference label needs a real reason, not a proxy.**
It would have been easy to just label every low-temperature response "chosen" — but temperature isn't quality. Routing every pair through an LLM judge, and dropping pairs it couldn't confidently decide, kept the dataset honest at the cost of some pairs being thrown away.

**DPO's memory cost comes from the reference model, not the LoRA adapter.**
DPO's loss compares the policy model against a frozen reference copy of itself, which has to stay resident throughout training. That's what actually strains memory — the LoRA adapter itself is small; keeping two copies of the base model in memory is the real constraint.

**Position bias in LLM judges is real and needs an explicit control.**
Judging each pair twice with the order swapped, and only crediting a win when a model wins both orderings, turned up cases where the two orderings disagreed — the textbook signature of position bias. Scoring those as ties instead of picking one was the only honest option.

**A win rate is falsifiable in a way a single example isn't.**
It would have been faster to just generate one example and say "look how much better the DPO model sounds." A win rate across held-out prompts, with the raw responses saved alongside it, is a claim someone else can actually check.

**Holding out prompts from both training stages mattered more than expected.**
Evaluating on prompts seen during either SFT or DPO training would have measured memorization, not generalization. The held-out set is what makes the win rate mean "the model's behavior actually shifted" instead of "the model remembers its training data."

---

## Related Projects

- [llm-fine-tuning](https://github.com/Honaxen/llm-fine-tuning) — the SFT model this project aligns further with DPO
- [llm-safety-redteam](https://github.com/Honaxen/llm-safety-redteam) — the same LLM-as-judge evaluation pattern, applied to preference alignment instead of safety

---

## Author

[Honaxen](https://github.com/Honaxen)