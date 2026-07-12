# Architecture

## Overview

This project takes an already instruction-tuned (SFT) model and aligns it
further to human preferences using DPO, then proves the alignment actually
changed behavior rather than just adding training time:

```
Preference Dataset (data/)
    |
    v
DPO Training (training/)               refine the SFT model on preference pairs
    |
    v
Head-to-Head Evaluation (evaluation/)  SFT-only vs SFT+DPO, judged on held-out prompts
    |
    v
Win Rate Report                        did alignment measurably change behavior?
```

Everything here builds on `llm-fine-tuning`'s SFT model rather than
starting from the raw base model -- DPO is a refinement step, not a
replacement for instruction-tuning. The evaluation stage exists because
"we ran DPO" isn't evidence of anything; a measured win rate against a
judge, on prompts the model never saw during either training stage, is.

---

## Stage 1: Preference Dataset

`data/build_preference_dataset.py` generates preference pairs via
self-play: for each seed prompt, the same base model produces two
responses at different sampling temperatures (0.3, focused; 1.1,
exploratory), and an LLM-as-judge decides which one is actually better
for that specific prompt.

Temperature alone isn't used as the preference signal -- a low-temperature
response isn't automatically "better," and sometimes the higher-temperature
response is more creative while remaining correct. That's exactly why a
judge call is in the loop instead of just labeling every low-temperature
response as chosen: the point is to capture genuine quality differences,
not sampling-parameter differences.

Pairs the judge can't confidently decide on are dropped rather than
guessed at -- a preference dataset with noisy labels teaches the model
the wrong lesson, so it's better to have fewer, cleaner pairs than more,
noisier ones.

---

## Stage 2: DPO Training

`training/train_dpo.py` loads the SFT LoRA adapter from `llm-fine-tuning`,
merges it into the base model, and trains a fresh LoRA adapter on top
using Direct Preference Optimization.

**Why DPO instead of full RLHF (reward model + PPO):** DPO reframes the
RLHF objective as a single loss computed directly on (chosen, rejected)
pairs, with no separate reward model to train and no PPO rollout loop.
It's simpler to implement correctly and has become the standard starting
point for preference alignment in practice, at the cost of some of the
flexibility a learned reward model provides.

**Why LoRA again, and why it matters more here than in SFT:** DPO's loss
requires comparing the policy model's output probabilities against a
frozen reference copy of the same model. That reference copy has to stay
in memory throughout training, which makes full fine-tuning even more
memory-hungry for DPO than it was for SFT. LoRA keeps this affordable on
the same hardware used for `llm-fine-tuning`.

The LoRA config (rank, target modules) is kept identical to the SFT
adapter's config on purpose -- the only variable that should change
between the SFT-only and SFT+DPO models is the training objective, not
the adapter's capacity.

---

## Stage 3: Head-to-Head Evaluation

`evaluation/compare_sft_vs_dpo.py` runs both models -- SFT-only and
SFT+DPO -- on a held-out prompt set that appears in neither the SFT
training data nor the DPO preference dataset, and has an LLM judge pick
the better response for each prompt.

**Position-bias control:** LLM judges have a documented tendency to
slightly favor whichever response is shown first. Each pair is judged
twice with the order swapped; a model only counts as the winner if it
wins under both orderings. Disagreement between orderings is scored a
tie rather than arbitrarily broken -- that disagreement is itself the
signature of position bias, not a real preference.

The output is a win rate, not a single example: "DPO won 68% of held-out
prompts" is a claim that can be checked and reproduced, unlike "look at
this one example where the DPO model sounds nicer."

---

## Why This Order

- DPO (Stage 2) needs a preference dataset (Stage 1) to train on -- there's
  no DPO loss without (chosen, rejected) pairs.
- Evaluation (Stage 3) needs both a trained DPO adapter and the original
  SFT adapter to compare against -- without the baseline, "68% win rate"
  has no meaning.
- Using held-out prompts, disjoint from both the SFT and DPO training
  data, is what separates "the model memorized these specific answers"
  from "the model's general behavior actually shifted."

The pipeline is meant to answer one specific, falsifiable question:
does DPO change what this model does, measurably, on prompts it has
never seen -- not just on the pairs it was trained on.