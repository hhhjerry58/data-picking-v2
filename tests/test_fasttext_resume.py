import subprocess
from pathlib import Path

from picking import _fasttext_native
from picking.common import file_hash, REPO


def arguments(data, output):
    return ["fasttext", "supervised", "-input", str(data), "-output", str(output), "-dim", "16", "-epoch", "3",
            "-lr", "0.1", "-wordNgrams", "2", "-minCount", "1", "-bucket", "1000", "-minn", "0", "-maxn", "0",
            "-loss", "softmax", "-thread", "1", "-seed", "42", "-verbose", "0"]


def test_native_equivalence_and_exact_mid_epoch_resume(tmp_path):
    data = tmp_path / "train.txt"
    lines = []
    for i in range(160):
        lines += [f"__label__include solve the equation x plus {i} equals {i+7} with algebra\n",
                  f"__label__exclude this is a fictional weather report about city number {i}\n"]
    data.write_text("".join(lines))
    args = arguments(data, tmp_path / "upstream")
    subprocess.run([str(REPO / "vendor/fasttext/fasttext"), *args[1:]], check=True, capture_output=True)
    full = _fasttext_native.Classifier()
    full.initialize(args)
    initial = full.input_stats()
    assert initial["nonfinite"] == 0
    assert initial["zeros"] < initial["elements"] * 0.001
    assert initial["absolute_max"] <= 1/16
    full.advance(3)
    full.save(str(tmp_path / "full.bin"), str(tmp_path / "full.state"))
    assert file_hash(tmp_path / "full.bin") == file_hash(tmp_path / "upstream.bin")
    partial = _fasttext_native.Classifier()
    partial.initialize(args)
    partial.advance(3, 137)
    assert not partial.status()["finished"]
    partial.save(str(tmp_path / "partial.bin"), str(tmp_path / "partial.state"))
    restored = _fasttext_native.Classifier()
    restored.initialize(args, str(tmp_path / "partial.bin"), str(tmp_path / "partial.state"))
    assert restored.status() == partial.status()
    restored.advance(3)
    restored.save(str(tmp_path / "restored.bin"), str(tmp_path / "restored.state"))
    assert restored.status() == full.status()
    assert file_hash(tmp_path / "restored.bin") == file_hash(tmp_path / "full.bin")
    assert file_hash(tmp_path / "restored.state") == file_hash(tmp_path / "full.state")
    probe = ["solve an algebra equation with x", "a weather report about a city"]
    assert restored.predict_include(probe) == full.predict_include(probe)
    assert full.predict_include(probe)[0] > full.predict_include(probe)[1]
    inference = _fasttext_native.Classifier()
    inference.load_model(str(tmp_path / "restored.bin"))
    assert inference.predict_include(probe) == full.predict_include(probe)
