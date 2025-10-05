# GenAI-Project-2025
Team members: Binesh Arakkal Remesh, Zahra Sharifi, Elise Soenen.

This project is part of the Generative AI course (2025). It consists of data-to-text generation task using the [Google T5-small model (60M)](https://huggingface.co/google-t5/t5-small) and the [E2E NLG dataset](https://gem-benchmark.com/data_cards/e2e_nlg), used through HuggingFace and GEM platform respectively.

How-to-run:
First of all, install the required libraries with
```
pip install -r requirements.txt
```
and then run the notebook *main.py*.

Content of the files:
- *1_corpus_utils.py* contains the code to download the corpus and the model.
- *2_data_preprocessing.py* contains the code with the encoder/tokenizer to preprocess the input data (no tokenization is applied, only truncation and linearization)
- *3_train.py* contains the code for the training phase of the model.
- *4_generation.py* contains the code with examples of text generation to observe the outputs.
- *5_evaluation.py* contains all the metrics used for the evaluation phase and their scores.
- *evaluation.csv* reports all the evaluation scores performed by the model (Bleu, Rouge-1, Rouge-2, Rouge-L, chrf++)
- And *main.py* file contains all the previous *\*.py* files.

NB: The file *main.py* was splited at each step for an easier debugging process. If the splited files are used instead of the main one, there are to run in the order of their indicated number.
