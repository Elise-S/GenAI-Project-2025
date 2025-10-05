import ctranslate2
from tqdm import tqdm

def convert_to_ctranslate(model_dir='t5-e2e_nlg', output_dir='t5-small-e2e_nlg-ct2'): # Convert the model to ctranslate format
    import os
    os.system(f"ct2-transformers-converter --model {model_dir} --output_dir {output_dir}") 
    translator = ctranslate2.Translator(output_dir, device='cuda')
    return translator

def generate_predictions(translator, tokenizer, test_ds, batch_size=32, column_name='t5-small_e2e_nlg'):  #using the finetuned model to generate predictions
    
    def pre_process(text):  # encodeing the input text 
        input_ids = tokenizer.encode(text)
        input_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        return input_ids  # directly using ids for translator

    def post_process(output): # decoding the output text
        output_tokens = output.hypotheses[0]
        output_ids = tokenizer.convert_tokens_to_ids(output_tokens)
        return tokenizer.decode(output_ids, skip_special_tokens=True)

    batch = [pre_process(text) for text in tqdm(test_ds['input'])] #making batches for efficiency of the model generation
    generated_texts = []

    for i in tqdm(range(0, len(batch), batch_size)):
        subbatch = batch[i:i + batch_size]
        output = translator.translate_batch(subbatch, max_batch_size=batch_size)
        generated_texts += [post_process(o) for o in output]

    test_ds = test_ds.add_column(column_name, generated_texts)
    return test_ds
