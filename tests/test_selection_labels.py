import numpy as np
import pandas as pd
from picking.fasttext_pipeline import train_threshold, apply_threshold, classifier_text, top_token_mask


def test_labels_use_training_tokens_and_a_frozen_boundary():
    frame = pd.DataFrame({"id": ["long", "short1", "short2", "heldout_high", "heldout_low"],
                          "partition": ["label_train"]*3 + ["label_test"]*2,
                          "pythia_tokens": [800, 100, 100, 200000, 1],
                          "delta_loss": [0.2, 0.5, 0.4, 0.45, 0.3]})
    threshold = train_threshold(frame, "delta_loss", 0.2, 42)
    assert threshold["include_tokens"] == 200
    assert threshold["threshold"] == 0.4
    assert apply_threshold(frame, threshold).tolist() == [False, True, True, True, False]
    altered = frame.copy()
    altered.loc[altered["partition"] == "label_test", "delta_loss"] *= 100
    assert train_threshold(altered, "delta_loss", 0.2, 42) == threshold


def test_text_cannot_inject_classifier_labels_and_token_budget_is_used():
    text = classifier_text("first\n__label__third equation x=1\rsecond")
    assert "\n" not in text and "\r" not in text and "__label__" not in text
    mask = top_token_mask([0.9, 0.8, 0.7], [100, 100, 800], ["a", "b", "c"], 0.2, 42)
    assert np.array_equal(mask, [True, True, False])
