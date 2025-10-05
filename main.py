from corpus_utils import load_e2e_dataset, simplify_train_dataset, simplify_validation_dataset, simplify_test_dataset
from data_preprocessing import tokenize_sample
from train import train_t5_model
from generation import convert_to_ctranslate, generate_predictions
from evaluation import evaluate_and_save_scores

def main():
    
    e2e_data = load_e2e_dataset() # loading the dataset

    # simplydy the train validation and test datasets
    train_ds = e2e_data['train'].map(simplify_train_dataset, batched=True, remove_columns=e2e_data['train'].column_names)
    valid_ds = e2e_data['validation'].map(simplify_validation_dataset, batched=True, remove_columns=e2e_data['validation'].column_names)
    test_ds = e2e_data['test'].map(simplify_test_dataset, batched=True, remove_columns=e2e_data['test'].column_names)

    # tokenizing the preprocessed datasets
    tokenized_train_ds = train_ds.map(tokenize_sample, batched=True, remove_columns=['input', 'graph', 'reference'], batch_size=2000)
    tokenized_valid_ds = valid_ds.map(tokenize_sample, batched=True, remove_columns=['input', 'graph', 'reference'], batch_size=2000)

  # finetuning the t5-small model for our dataset 
    model, tokenizer = train_t5_model(tokenized_train_ds, tokenized_valid_ds)


    translator = convert_to_ctranslate() # converting into ctranslate2 format for faster generation less gpu memory usage


    test_ds = generate_predictions(translator, tokenizer, test_ds) # geneating the predictions of the dataset test


    json_file = 't5_small_e2e_nlg_gen-response.json'  #  saving the  predictions into JSON
    test_ds.to_json(json_file)

    # Step 7: Evaluate and save scores
    csv_file = "evaluation_scores.csv"
    evaluate_and_save_scores(json_file, csv_file)


if __name__ == "__main__":
    main()
