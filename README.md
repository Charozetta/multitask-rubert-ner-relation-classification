# Multi-task RuBERT for Russian NER and Relation-Type Classification

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/3906228f-5cd9-4b7e-bbba-bb42e78f67a3" />


This project trains a compact shared **RuBERT** encoder for two complementary information-extraction tasks on Russian news texts:

1. **Named entity recognition (NER):** assigning BIO entity labels to tokens.
2. **Document-level multi-label relation-type classification:** predicting which of 30 relation types are present in a document.

The implementation is a reproducible educational research prototype. A shared Transformer encoder feeds a token-classification head and a document-classification head, allowing both tasks to be trained jointly.

> **Scope:** This project predicts token-level entity labels and document-level relation-type presence. It does **not** identify entity pairs or extract relation triples; it is therefore not a full relation-extraction system.

## Why Multi-task Learning?

NER and relation-type classification draw on overlapping contextual signals: entities, event descriptions, roles, locations, dates, and organizations. The project tests whether a single encoder can support both tasks while retaining task-specific output heads.

The model uses `cointegrated/rubert-tiny2` as the shared encoder, a linear BIO token-classification head, and a linear document-level multi-label classification head.

## Dataset

The notebook downloads the public `danasone/nerel` dataset directly from the Hugging Face Hub. The data is derived from **NEREL**, a Russian information-extraction resource with nested entities, relations, events, and entity links. The original collection documents 29 entity types and 49 relation types.

This experiment uses the dataset's available word-level BIO labels and a 30-dimensional multi-hot document target. No dataset files are included in the repository.

> **Data notice:** Download the dataset from its source and comply with the terms and attribution requirements of the exact Hugging Face dataset version used. This repository contains code and documentation only.

## Methodology

The notebook tokenizes pre-segmented words with a fast tokenizer, transfers every word-level BIO label to its first subword, and assigns `-100` to padding and subsequent subwords. This excludes non-evaluated positions from the token-classification loss.

The public workflow uses a fixed-seed split of approximately **70% training, 15% validation, and 15% untouched test data**. The validation split is used for model selection, early stopping, and document-label threshold tuning. The final test split is evaluated only after the checkpoint and thresholds are frozen.

| Component | Design choice |
|---|---|
| Shared encoder | `cointegrated/rubert-tiny2` |
| Token task | BIO token classification |
| Document task | 30-label multi-label relation-type presence classification |
| Token loss | Class-weighted cross-entropy with `ignore_index=-100` |
| Document loss | `BCEWithLogitsLoss` with training-split positive-class weights |
| Multi-task weighting | Learnable uncertainty weighting |
| Optimizer | AdamW with linear warmup and decay |
| Model selection | Mean of validation entity-span F1 and document micro F1 |
| Final evaluation | One evaluation on the untouched test split |

The class-aware losses respond to label imbalance using weights calculated from the **training split only**. They are modelling choices rather than guaranteed improvements, so their impact is assessed on validation data.

## Metrics

NER metrics are intentionally separated to avoid ambiguity:

| Metric | Interpretation |
|---|---|
| BIO token macro F1, all labels | Macro F1 over all BIO tag IDs, including the background `O` label. |
| BIO token macro F1, entity labels only | Macro F1 over non-`O` BIO labels. |
| Entity-span F1 | Sequence-level F1 calculated with `seqeval` on reconstructed BIO sequences. |
| Document micro F1 | Micro F1 over all 30 document-level labels after validation-selected per-label thresholding. |
| Document macro F1 | Macro F1 over document-level labels; more sensitive to rare relation types. |

> **Important:** The notebook deliberately does not display the metrics from the earlier coursework version. That version repeatedly used the held-out split for model selection and therefore produced validation estimates, not independent final-test estimates. Run this public version to obtain methodologically clean final metrics.

## Repository Structure

```text
multitask-rubert-ner-relation-classification/
├── notebooks/
│   └── multitask_rubert_ner_relation_classification.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## Run the Project

The notebook can be run in Google Colab, JupyterLab, or Jupyter Notebook. A GPU is recommended but not required.

```bash
git clone https://github.com/<your-github-username>/multitask-rubert-ner-relation-classification.git
cd multitask-rubert-ner-relation-classification
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/multitask_rubert_ner_relation_classification.ipynb
```

Run the notebook from top to bottom. It downloads the dataset and pretrained RuBERT weights automatically on the first run. The training cell writes `best_multitask_rubert.pt` locally; this artifact is intentionally excluded from version control.

## Limitations and Next Steps

The model is a useful multi-task baseline, but it has several constraints. The dataset is small for a Transformer fine-tuning experiment; the model uses a compact encoder; word-level BIO conversion does not represent NEREL's nested-entity structure fully; and document-level relation-type presence is a weaker task than entity-pair relation extraction.

Appropriate future work includes comparing larger Russian encoders, nested-NER methods, alternative task-weighting strategies, cross-validation or repeated splits, calibration of document-level probabilities, and a relation head that explicitly predicts relations between detected entity pairs.


