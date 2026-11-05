import math
from picking.selection import input_tokens_for_targets


def test_exact_prediction_budget_for_boundary_lengths():
    for context in [4, 17, 2048]:
        for target in [1, context-1, context, context+1, 100000, 3200000000]:
            inputs = input_tokens_for_targets(target, context)
            blocks = math.ceil(inputs/context)
            assert inputs-blocks == target
    assert input_tokens_for_targets(3200000000, 2048) == 3201563264
