from picking.math_answers import extract_answer,equivalent


def test_boxed_nested_fraction_and_safe_numeric_equivalence():
    assert extract_answer('Earlier \\boxed{0}. Finally \\boxed{\\frac{1}{2}}.')=='\\frac{1}{2}'
    assert equivalent('0.5','\\frac{1}{2}')
    assert equivalent('\\frac12','\\dfrac{1}{2}')
    assert equivalent('x = 2','2')
    assert not equivalent('3','2')
    assert extract_answer('Reasoning without an answer.') is None
    assert not equivalent("__import__('os').system('x')",'0')
