import json
import evaluate
import pandas as pd

def evaluate_and_save_scores(json_file: str, output_csv: str):
    """
    this function is defined to evaluate the responses of the finetuned model
    scores are  CHRF++, BLEU, METEOR, and ROUGE, and save the scores to a CSV file.
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    predictions = data['t5-small_e2e_nlg']
    references = data['reference']

    max_refs = max([len(ref) if isinstance(ref, list) else 1 for ref in references])
    padded_references = []
    for ref in references:
        if isinstance(ref, list):
            temp = ref.copy()
        else:
            temp = [ref]
        while len(temp) < max_refs:
            temp.append('')
        padded_references.append(temp)


    chrf = evaluate.load('chrf') # chrf is loded here
    bleu = evaluate.load("bleu") # bleu score is loaded here
    meteor = evaluate.load("meteor") # meteor score is loaded here
    rouge = evaluate.load("rouge") # rouge score is loaded here

    # all the scores are computed in the following codes
    chrf_score = chrf.compute(predictions=predictions, references=padded_references, word_order=2)
    bleu_score = bleu.compute(predictions=predictions, references=padded_references)
    meteor_score = meteor.compute(predictions=predictions, references=padded_references)
    rouge_score = rouge.compute(predictions=predictions, references=padded_references)

#results are saved in a dict to save it as a csv file
    results = {
        'CHRF++': chrf_score['score'],
        'BLEU': bleu_score['bleu'],
        'METEOR': meteor_score['meteor'],
        'ROUGE-1': rouge_score['rouge1'],
        'ROUGE-2': rouge_score['rouge2'],
        'ROUGE-L': rouge_score['rougeL']
    }

# results are saving as a csv file 
    df = pd.DataFrame([results])
    df.to_csv('evaluation_scores.csv', index=False)
    print(f"Evaluation scores saved to evaluation_scores.csv")
    print(results)
