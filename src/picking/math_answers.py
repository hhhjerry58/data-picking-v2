"""Deterministic string/rational grading; never execute generated expressions."""
import re
import unicodedata
from fractions import Fraction


def extract_answer(text):
    matches = list(re.finditer(r"\\(?:boxed|fbox)\s*\{", text))
    if matches:
        start = matches[-1].end()
        depth = 1
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:index].strip() or None
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    explicit = re.findall(r"(?:final\s+answer|the\s+answer|answer)\s*(?:is|:|=)\s*([^\n]+)", text, re.I)
    if explicit:
        return explicit[-1].strip().rstrip(".")
    if lines and re.fullmatch(r"\$?[-+\d\s,./%]+\$?\.?", lines[-1]):
        return lines[-1].strip().rstrip(".")
    return None


def normalize_answer(value):
    text = unicodedata.normalize("NFKC", value).strip().rstrip(".")
    text = text.replace("$", "").replace("\\left", "").replace("\\right", "")
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = re.sub(r"\\(?:!|,|;|quad|qquad)", "", text)
    text = re.sub(r"\\(?:text|mathrm)\{\s*(?:cm|mm|meters?|square units|units|degrees)\s*\}", "", text)
    text = text.replace("\\%", "%").replace("\\{", "{").replace("\\}", "}").replace("π", "\\pi")
    text = re.sub(r"\\sqrt([a-zA-Z0-9])", r"\\sqrt{\1}", text)
    text = re.sub(r"\\frac([+-]?\d)\{([^{}]+)\}", r"\\frac{\1}{\2}", text)
    text = re.sub(r"\\frac\{([^{}]+)\}([+-]?\d)", r"\\frac{\1}{\2}", text)
    text = re.sub(r"\\frac([+-]?\d)(\d)", r"\\frac{\1}{\2}", text)
    text = re.sub(r"\s+", "", text)
    if re.match(r"^[a-zA-Z]=", text):
        text = text.split("=", 1)[1]
    return text


def rational(value):
    if len(value) > 160:
        return None
    match = re.fullmatch(r"\\frac\{([+-]?\d+)\}\{([+-]?\d+)\}", value)
    try:
        if match:
            return Fraction(int(match[1]), int(match[2]))
        if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:/[+-]?\d+)?", value):
            return Fraction(value)
    except (ValueError, ZeroDivisionError):
        pass
    return None


def equivalent(prediction, reference):
    if prediction is None or reference is None:
        return False
    left, right = normalize_answer(prediction), normalize_answer(reference)
    if left == right:
        return True
    a, b = rational(left), rational(right)
    return a is not None and b is not None and a == b
