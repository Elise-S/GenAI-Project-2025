# GenAI-Project-2025
Team members: Binesh Arakkal Remesh, Zahra Sharifi, Elise Soenen.

This project is part of the Generative AI course (2025). It consists of data-to-text generation task using the [Google T5-small model (60M)](https://huggingface.co/google-t5/t5-small) and the [E2E NLG dataset](https://gem-benchmark.com/data_cards/e2e_nlg), used through HuggingFace and GEM platform respectively.


### How-to-run:
First of all, install the required libraries with
```
pip install -r requirements.txt
```
and then run the notebook *main.py*.


### Content of the files:
- Folder Code contain :- 
  - *corpus_utils.py* -> download the corpus and some of the preprocessing steps (no tokenization is applied, only cleaning and linearization fonctions).
  - *data_preprocessing.py* -> contains the code with the encoder/tokenizer to further preprocess the input data (with tokenization applied).
  - *train.py* -> training phase of the model.
  - *generation.py* contains the code to generate the fine-tuned model answers.
  - *evaluation.py* contains all the metrics used for the evaluation phase and their scores.
  - And *main.py*, the file to run, contains fonctions to run all the previous *\*.py* files.
- Folder notebook contain :-
  - *GenAI-project_meaning_representation.ipynb* is the notebook gathers all the code with our outputs.
- Folder results contain :-
  - *evaluation.csv* reports all the evaluation scores performed by the model (Bleu, Rouge-1, Rouge-2, Rouge-L, chrf++).
  - *t5_small_after_fine_tuning.json* which contain the generated output of the model after finetuned on E2E NLG corpus
  - *t5_small_before_fine_tuning.json* contain generated output of the model before finetuning.
