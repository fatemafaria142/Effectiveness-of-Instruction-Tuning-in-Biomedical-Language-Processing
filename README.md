# Effectiveness of Instruction Tuning in Biomedical Language Processing

## Overview

This project explores the effectiveness of instruction tuning in biomedical language processing using the Llama2-MedTuned-Instructions dataset. The dataset is designed for training language models in biomedical NLP tasks, containing approximately 200,000 samples with specific instructions for tasks such as Named Entity Recognition (NER), Relation Extraction (RE), and Medical Natural Language Inference (NLI).

## Dataset

- **Dataset Name:** Llama2-MedTuned-Instructions
- **Dataset Link:** [Llama2-MedTuned-Instructions Dataset](https://huggingface.co/datasets/nlpie/Llama2-MedTuned-Instructions?row=0)
- **Description:** The Llama2-MedTuned-Instructions dataset is tailored for instruction-based learning, providing guidance to language models for various biomedical NLP tasks.

## Models Used
1. **Llama-2-7b-chat-hf**
   - **Model Link:** [Llama-2-7b-chat-hf Model](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
   - **Description:**   Llama 2 is an auto-regressive language model that uses an optimized transformer architecture. The tuned versions use supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) to align to human preferences for helpfulness and safety.

2. **Mistral-7B-v0.1**
   - **Model Link:** [Mistral-7B-v0.1 Model](https://huggingface.co/mistralai/Mistral-7B-v0.1)
   - **Description:**  Mistral-7B-v0.1 Large Language Model (LLM), a cutting-edge pretrained generative text model boasting an impressive 7 billion parameters. This powerhouse outshines the competition, surpassing the benchmarks set by Llama 2 13B across all tested metrics.
     
3. **TinyLlama**
   - **Model Link:** [TinyLlama-1.1B-Chat-v1.0  Model](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
   - **Description:** The TinyLlama project aims to pretrain a 1.1B Llama model on 3 trillion tokens. 

4. **Starling-LM-7B-alpha**
   - **Model Link:** [Starling-LM-7B-alpha Model](https://huggingface.co/berkeley-nest/Starling-LM-7B-alpha)
   - **Description:** Starling-7B, an open large language model (LLM) trained by Reinforcement Learning from AI Feedback (RLAIF). 

5. **Mistral-7B-Instruct-v0.2**
   - **Model Link:** [Mistral-7B-Instruct-v0.2  Model](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
   - **Description:** The Mistral-7B-Instruct-v0.2 Large Language Model (LLM) is an improved instruct fine-tuned version of Mistral-7B-Instruct-v0.1.
  
6. **OpenChat 3.5 1210**
   - **Model Link:** [OpenChat 3.5 1210  Model](https://huggingface.co/openchat/openchat-3.5-0106)
   - **Description:** OpenChat is an innovative library of open-source language models, fine-tuned with C-RLFT - a strategy inspired by offline reinforcement learning.

7. **GPT2**
   - **Model Link:** [GPT2 Documentation](https://huggingface.co/docs/transformers/model_doc/gpt2)
   - **Description:** GPT2 serves as one of the language models used in this project. It is fine-tuned on the Llama2-MedTuned-Instructions dataset for biomedical language processing tasks.

8. **GPT-Medium**
   - **Model Link:** [GPT-Medium Model](https://huggingface.co/openai-community/gpt2-medium)
   - **Description:** GPT-Medium, a variant of GPT2, is also employed in this project. GPT-2 Medium is the 355M parameter version of GPT-2, a transformer-based language model created and released by OpenAI. The model is pretrained on the English language using a causal language modeling (CLM) objective, enabling it to generate coherent and contextually relevant text based on given prompts. It is well-suited for various natural language processing tasks, making it a valuable asset for biomedical language understanding in this project.
     



