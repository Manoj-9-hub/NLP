from backend.app.services.evaluation import calculate_rouge

def test_rouge_evaluation():
    generated = "T5 model summarizes long documents."
    reference = "T5 model summarizes long documents."
    
    # Perfect match test
    res = calculate_rouge(generated, reference)
    assert res.rouge1.fmeasure == 1.0
    assert res.rouge2.fmeasure == 1.0
    assert res.rougeL.fmeasure == 1.0
    assert res.original_length == 5
    assert res.generated_length == 5
    
    # Partial match test
    gen_partial = "T5 model generates summaries."
    res_partial = calculate_rouge(gen_partial, reference)
    assert res_partial.rouge1.fmeasure > 0.0
    assert res_partial.rouge1.fmeasure < 1.0
