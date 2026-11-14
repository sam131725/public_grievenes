import pytest

from grievance.data.preprocess import clean_text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Please check https://example.com/ticket?id=4 soon", "Please check soon"),
        ("Report at WWW.example.org/path", "Report at"),
        ("  Water   is\nnot   available  ", "Water is not available"),
    ],
)
def test_clean_text_removes_links_and_normalizes_whitespace(text, expected):
    assert clean_text(text) == expected
