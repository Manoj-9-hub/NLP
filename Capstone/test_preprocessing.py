from backend.app.services.preprocessing import clean_text, detect_sentences

def test_clean_text():
    # Test spaces normalization
    assert clean_text("  This   is a    spaced   sentence.  ") == "This is a spaced sentence."
    # Test blank inputs
    assert clean_text("") == ""

def test_detect_sentences():
    # Test multiple sentence extraction
    text = "The T5 model treats NLP tasks as text-to-text. This is a second sentence! And a third?"
    sentences = detect_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "The T5 model treats NLP tasks as text-to-text."
    assert sentences[1] == "This is a second sentence!"
    assert sentences[2] == "And a third?"

    # Test abbreviation checks (should not split on Dr. or e.g.)
    text_with_abbrev = "Dr. Smith went to the lab, e.g., the biotechnology lab. It was great."
    sents = detect_sentences(text_with_abbrev)
    assert len(sents) == 2
    assert sents[0] == "Dr. Smith went to the lab, e.g., the biotechnology lab."
    assert sents[1] == "It was great."
