from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

def train_t5_model(tokenized_train_ds, tokenized_valid_ds, output_dir='t5-e2e_nlg'):
    """
    fine_tuning function of T5 model after tokenize.
    """
    model_name = "google/t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3, 
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,
        eval_strategy='steps',
        eval_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        bf16=True,
        optim='adafactor',
        report_to=[]
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_ds,
        eval_dataset=tokenized_valid_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer)
    )

    trainer.train()
    trainer.model.save_pretrained(output_dir)  # saving the fine-tuned model
    tokenizer.save_pretrained(output_dir)

    return model, tokenizer
