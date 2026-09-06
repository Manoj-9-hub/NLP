from backend.app.services.chunking import create_chunks, estimate_tokens

def test_estimate_tokens():
    text = "The T5 model represents everything as text"
    # 7 words * 1.3 = 9.1 -> 9 estimated tokens
    assert estimate_tokens(text) == 9
    assert estimate_tokens("") == 0

def test_create_chunks():
    # Write representative sentences
    sentences = [
        "This is sentence one containing words.",
        "And this is sentence two which adds more words here.",
        "Finally sentence three is also included in this package."
    ]
    # If we request chunk_size of 15 words and 3 overlap words:
    # Sentence 1: 6 words
    # Sentence 2: 10 words
    # Both: 16 words (exceeds 15) -> Sentence 1 should go to Chunk 1, Sentence 2 starts Chunk 2
    chunks = create_chunks(sentences, chunk_size=15, chunk_overlap=3)
    assert len(chunks) >= 2
    assert "sentence one" in chunks[0]
    assert "sentence two" in chunks[1]
