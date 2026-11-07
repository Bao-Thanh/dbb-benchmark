"""Real TinyLlama response examples reported by a user (synthetic test text only)."""
import pytest
from dbb_metrics import parse_choice,interpret
from dbb_compact_worker import build_compact_dbb_prompt


@pytest.mark.parametrize("raw,answer,reason", [
    ("Option: A", "A", "single_explicit_choice"),
    ("Option B", "B", "single_explicit_choice"),
    ("Answer: Option B", "B", "single_explicit_choice"),
    ("Option: A (only one letter is available)\nExplanation: Option A is a continuation. On the other hand, option B is a continuation.", "A", "explicit_choice_with_extra_text"),
    ("Option B is correct as available. The continuation is descriptive.", "B", "explicit_choice_with_extra_text"),
    ("Option: (C)\nReturning exactly one option letter (C).", "INVALID", "prompt_echo"),
    ("Return exactly one option letter (A, B or C as available); no explanation.", "INVALID", "prompt_echo"),
    ("Option: A\nReturn exactly one option letter (A, B or C as available); no explanation.", "INVALID", "prompt_echo"),
    ("Option: A or B", "INVALID", "multiple_explicit_choices"),
    ("Option: A\nFinal answer: B", "INVALID", "multiple_explicit_choices"),
    ("A\nB", "INVALID", "multiple_explicit_choices"),
    ("I choose A", "INVALID", "ambiguous_or_nonconforming_output"),
    ("Option: C", "INVALID", "letter_not_offered"),
    ("B. She greeted each customer with a genuine smile.", "B", "explicit_choice_with_extra_text"),
    ("I cannot choose", "INVALID", "ambiguous_or_nonconforming_output"),
])
def test_tinyllama_examples(raw,answer,reason):
    assert parse_choice(raw,'AB')==(answer,reason)


def test_mapping_preserved_for_recoverable_output():
    assert interpret('Option: A\nExplanation: B is not correct', {'A':'ans2','B':'ans1'})['selected_original']=='ans2'


def test_compact_prompt_has_no_c_option_or_stale_instruction():
    prompt=build_compact_dbb_prompt('some context',{'A':'foo','B':'bar'})
    assert 'ONLY A or B' in prompt
    assert 'C as available' not in prompt
    assert 'A. foo' in prompt and 'B. bar' in prompt
