# LLM Preference Alignment (DPO)

Work in progress -- this README is a placeholder and will be replaced once the project is complete.

Aligning an instruction-tuned LLM to human preferences using Direct Preference Optimization (DPO) -- and measuring exactly how much that changes behavior compared to instruction-tuning alone.

---

## What This Project Will Demonstrate

llm-fine-tuning showed how to teach a model to follow instructions (SFT).
This one shows how to teach a model which of two correct-ish answers is actually better -- the step that turns an instruction-follower into something that behaves like a real assistant.

Concern -> Solution (planned)
- How do you teach "better", not just "correct"?   -> Preference dataset: prompt + chosen response + rejected response
- How do you train on preferences directly?         -> Direct Preference Optimization (DPO), no separate reward model or PPO loop
- Did alignment actually change behavior?            -> Before/after comparison: SFT-only vs SFT+DPO, judged head-to-head
- Is the improvement real or just noise?              -> Win rate against a judge, not a single eyeballed example

---

## Planned Architecture

Preference Dataset (data/)                    prompt + chosen + rejected triples
  -> DPO Training (training/)                  fine-tune the SFT model directly on preference pairs
  -> Evaluation (evaluation/)                  SFT-only vs SFT+DPO, head-to-head LLM-as-judge comparison
  -> Win Rate Report                            % of prompts where DPO output is preferred

---

## Project Structure

llm-preference-alignment/
  data/            - preference dataset (prompt, chosen, rejected)
  training/        - DPO training script
  evaluation/       - before/after head-to-head comparison
  tests/
  docs/

---

## Stack

Python - PyTorch - PEFT - TRL - HuggingFace - Ollama

---

## Status

- [ ] Preference dataset (prompt/chosen/rejected triples)
- [ ] DPO training script
- [ ] SFT-only vs SFT+DPO head-to-head evaluation
- [ ] Win rate report

---

## Related Projects

- [llm-fine-tuning](https://github.com/Honaxen/llm-fine-tuning) -- the SFT model this project aligns further with DPO
- [llm-safety-redteam](https://github.com/Honaxen/llm-safety-redteam) -- same LLM-as-judge evaluation pattern, applied to preference alignment instead of safety

---

## Author

[Honaxen](https://github.com/Honaxen)
