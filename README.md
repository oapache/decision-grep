# Decision-Grep

**Our fine-tuned code-relevance model for repository search.** Decision-Grep evaluates whether a candidate file or code excerpt contains concrete evidence that helps answer a developer's query. It is designed to rank a small candidate set gathered by ordinary search tools; it does not replace indexing or search itself.

## The model

Decision-Grep is a separate fine-tune of our Decizion Router checkpoint `router-v8-cal5`, itself based on [Laya multilingual](https://huggingface.co/convaiinnovations/laya-multilingual). It retains the same non-autoregressive mmBERT architecture with **322 million parameters** and is trained to score code relevance rather than route tasks. The dedicated checkpoint is `decision-grep-v1`.

Version 1 was trained using **100% synthetic data**: 360 generated examples across 12 topics, split into 192 training and 168 validation examples. The checkpoint file is approximately **1.29 GB**; inference is configured for local CUDA/FP16 serving. We plan to publish the fine-tuned weights on Hugging Face soon. They are not included in this Git repository.

## What it does

- Scores candidate code against a query using typed relevance questions.
- Integrates with [Decizion Router](https://github.com/oapache/decizion-router) through its `/v1/systemone` endpoint, using the `decision-grep-v1` model selector.
- Includes the synthetic-data training pipeline and a small synthetic training/validation dataset. No project repositories or private code are included.

## Evaluation note

The included examples are synthetic. Results on synthetic holdouts do not establish production accuracy on arbitrary repositories; validate on your own retrieval tasks before relying on rankings.

## Status

Source release is available here. The fine-tuned weights and complete setup instructions will follow on Hugging Face.
