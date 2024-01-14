# Effectiveness of Instruction Tuning in Biomedical Language Processing

## Overview

This project explores the effectiveness of instruction tuning in biomedical language processing using the Llama2-MedTuned-Instructions dataset. The dataset is designed for training language models in biomedical NLP tasks, containing approximately 200,000 samples with specific instructions for tasks such as Named Entity Recognition (NER), Relation Extraction (RE), and Medical Natural Language Inference (NLI).

## Dataset

- **Dataset Name:** Llama2-MedTuned-Instructions
- **Dataset Link:** [Llama2-MedTuned-Instructions Dataset](https://huggingface.co/datasets/nlpie/Llama2-MedTuned-Instructions?row=0)
- **Description:** The Llama2-MedTuned-Instructions dataset is tailored for instruction-based learning, providing guidance to language models for various biomedical NLP tasks.

## Models Used

1. **GPT2**
   - **Model Link:** [GPT2 Documentation](https://huggingface.co/docs/transformers/model_doc/gpt2)
   - **Description:** GPT2 serves as one of the language models used in this project. It is fine-tuned on the Llama2-MedTuned-Instructions dataset for biomedical language processing tasks.

2. **GPT-Medium**
   - **Model Link:** [GPT-Medium Model](https://huggingface.co/openai-community/gpt2-medium)
   - **Description:** GPT-Medium, a variant of GPT2, is also employed in this project. GPT-2 Medium is the 355M parameter version of GPT-2, a transformer-based language model created and released by OpenAI. The model is pretrained on the English language using a causal language modeling (CLM) objective, enabling it to generate coherent and contextually relevant text based on given prompts. It is well-suited for various natural language processing tasks, making it a valuable asset for biomedical language understanding in this project.
     
3. **Mistral 7B**
   - **Model Link:** [Mistral-7B-v0.1 Model](https://huggingface.co/mistralai/Mistral-7B-v0.1)
   - **Description:**  Mistral-7B-v0.1 Large Language Model (LLM), a cutting-edge pretrained generative text model boasting an impressive 7 billion parameters. This powerhouse outshines the competition, surpassing the benchmarks set by Llama 2 13B across all tested metrics.
     
4. **TinyLlama**
   - **Model Link:** [TinyLlama-1.1B-Chat-v1.0  Model](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
   - **Description:** The TinyLlama project aims to pretrain a 1.1B Llama model on 3 trillion tokens. 
   


