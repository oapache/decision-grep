# Decision-Grep

**A learned code-relevance scorer for repository search.** Decision-Grep evaluates whether a candidate file or code excerpt contains concrete evidence that helps answer a developer's query. It is designed to rank a small candidate set gathered by ordinary search tools; it does not replace indexing or search itself.

## What it does

- Scores candidate code against a query using typed relevance questions.
- Integrates with [Decizion Router](https://github.com/oapache/decizion-router) through its `/v1/systemone` endpoint, using the `decision-grep-v1` model selector.
- Includes the synthetic-data training pipeline and a small synthetic training/validation dataset. No project repositories or private code are included.

## Model availability

The training and evaluation code is published first. **The trained model weights are planned for Hugging Face soon.** Checkpoints are intentionally not stored in this Git repository. The training pipeline also needs the Decizion base checkpoint; setup and inference instructions will follow when those model files are available.

## Evaluation note

The included examples are synthetic. Results on synthetic holdouts do not establish production accuracy on arbitrary repositories; validate on your own retrieval tasks before relying on rankings.

## Status

Early source release. Model artifacts and complete setup instructions will follow on Hugging Face.
