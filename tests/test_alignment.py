"""
Unit tests for the judging logic in this project -- specifically the
parts that combine judge verdicts into a final decision. The actual
Ollama calls (call_ollama, call_judge) are mocked out, since testing
those would require a running model; what's being verified here is the
decision logic around them: does position-bias-controlled judging behave
correctly, and does the preference pair builder pick the right response.

Same split used throughout this portfolio: pure decision logic gets unit
tests, live-model calls are a manual/integration concern.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))

import build_preference_dataset  # noqa: E402
import compare_sft_vs_dpo  # noqa: E402


# --- build_preference_dataset.build_pair() tests ---

@patch("build_preference_dataset.judge_pair")
@patch("build_preference_dataset.call_ollama")
def test_build_pair_picks_response_a_as_chosen(mock_call_ollama, mock_judge_pair):
    mock_call_ollama.side_effect = ["focused response", "exploratory response"]
    mock_judge_pair.return_value = {"winner": "A", "reasoning": "more accurate"}

    pair = build_preference_dataset.build_pair("some-model", "judge-model", "a test prompt")

    assert pair["chosen"] == "focused response"
    assert pair["rejected"] == "exploratory response"


@patch("build_preference_dataset.judge_pair")
@patch("build_preference_dataset.call_ollama")
def test_build_pair_picks_response_b_as_chosen(mock_call_ollama, mock_judge_pair):
    mock_call_ollama.side_effect = ["focused response", "exploratory response"]
    mock_judge_pair.return_value = {"winner": "B", "reasoning": "more creative and still correct"}

    pair = build_preference_dataset.build_pair("some-model", "judge-model", "a test prompt")

    assert pair["chosen"] == "exploratory response"
    assert pair["rejected"] == "focused response"


@patch("build_preference_dataset.judge_pair")
@patch("build_preference_dataset.call_ollama")
def test_build_pair_returns_none_when_judge_undecided(mock_call_ollama, mock_judge_pair):
    mock_call_ollama.side_effect = ["response a", "response b"]
    mock_judge_pair.return_value = {"winner": None, "reasoning": "malformed output"}

    pair = build_preference_dataset.build_pair("some-model", "judge-model", "a test prompt")

    assert pair is None


# --- compare_sft_vs_dpo.judge_pair_both_orders() tests ---

@patch("compare_sft_vs_dpo.call_judge")
def test_dpo_wins_when_it_wins_both_orderings(mock_call_judge):
    # Order A (sft, dpo): winner "2" means dpo won
    # Order B (dpo, sft): winner "1" means dpo won
    mock_call_judge.side_effect = ["2", "1"]

    winner = compare_sft_vs_dpo.judge_pair_both_orders("judge-model", "prompt", "sft resp", "dpo resp")
    assert winner == "dpo"


@patch("compare_sft_vs_dpo.call_judge")
def test_sft_wins_when_it_wins_both_orderings(mock_call_judge):
    # Order A (sft, dpo): winner "1" means sft won
    # Order B (dpo, sft): winner "2" means sft won
    mock_call_judge.side_effect = ["1", "2"]

    winner = compare_sft_vs_dpo.judge_pair_both_orders("judge-model", "prompt", "sft resp", "dpo resp")
    assert winner == "sft"


@patch("compare_sft_vs_dpo.call_judge")
def test_result_is_tie_when_orderings_disagree(mock_call_judge):
    # Order A says dpo wins, Order B says sft wins -- classic position bias signature
    mock_call_judge.side_effect = ["2", "2"]  # order A: dpo wins; order B: sft wins

    winner = compare_sft_vs_dpo.judge_pair_both_orders("judge-model", "prompt", "sft resp", "dpo resp")
    assert winner == "tie"


@patch("compare_sft_vs_dpo.call_judge")
def test_result_is_tie_when_judge_call_fails(mock_call_judge):
    mock_call_judge.side_effect = [None, None]

    winner = compare_sft_vs_dpo.judge_pair_both_orders("judge-model", "prompt", "sft resp", "dpo resp")
    assert winner == "tie"
