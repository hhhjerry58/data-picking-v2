import numpy as np
import torch
from picking.scoring import document_windows, per_example_nll
from picking.data import TargetOverlap


def test_window_coverage_and_masked_causal_nll():
    ids = list(range(17))
    windows = list(document_windows(ids, 6))
    covered = [start+i for _, start, window in windows for i in range(1, len(window))]
    assert covered == list(range(1, len(ids)))
    tokens = torch.tensor([[0, 1, 2, 0], [1, 0, 0, 0]])
    mask = torch.tensor([[False, True, True, False], [False, True, False, False]])
    logits = torch.zeros(2, 4, 3)
    logits[0, 0, 1] = 2
    sums, counts = per_example_nll(logits, tokens, mask, chunk_size=1)
    expected_first = np.log(np.exp(2)+2)-2 + np.log(3)
    assert counts.tolist() == [2, 1]
    assert np.allclose(sums.numpy(), [expected_first, np.log(3)], atol=1e-6)
    logits[0, 2:] = 999
    logits[1, 1:] = 999
    repeated, _ = per_example_nll(logits, tokens, mask)
    assert torch.equal(sums, repeated)


def test_target_quarantine_uses_long_passages_not_single_math_words():
    targets = [{"id": "protected", "prompt": "Find the positive integer x such that the sum of x and seven equals nineteen.",
                "response": "Start by subtracting seven from both sides of the given equation. The resulting integer is twelve and it satisfies all of the requirements in the problem."}]
    checker = TargetOverlap(targets, {"overlap_problem_min_tokens": 12, "overlap_solution_min_tokens": 20, "overlap_containment": 0.8})
    assert checker.matches("An unrelated introduction. " + targets[0]["prompt"] + " More web content.")
    assert checker.matches("\n" + targets[0]["response"].upper() + "\n")
    assert not checker.matches("A different mathematical explanation about adding positive integers and solving equations.")
