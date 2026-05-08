# Pixels to Predictions: DL Vision Challenge

Pixels to Predictions: DL Vision Challenge is the final course competition for enrolled students, designed to evaluate your ability to build and rigorously assess a deep learning vision system for scientific multiple-choice reasoning. The goal is to develop a SmolVLM-500M-Instruct model (from an official Hugging Face pretrained checkpoint) that uses the provided images and question context to predict the correct answer choice for each test example, with leaderboard performance measured by prediction accuracy.

This competition is governed by strict academic constraints: teams may include up to 2 students, only the provided competition data may be used (no external data), evaluation runs in an offline setting with no internet access, development must remain within Google Colab Free tier and Kaggle Free tier resources, and the maximum number of trainable parameters is capped at 5 million. Participants should start from the provided starter notebook and follow Kaggle guidance here: Getting Started Guide, Competition Documentation, and Submitting Predictions.

## Start

13 days ago

## Close

2 days to go

## Evaluation

Submissions are scored using classification accuracy on a hidden test set. For each test example, your model must predict a single 0-indexed answer choice.

where N is the number of test examples, $\hat{y}_i$ is your predicted answer index, and $y_i$ is the ground-truth answer index.

The leaderboard is split into:

Public leaderboard: computed on a public subset of hidden test labels during the competition.

Private leaderboard: computed on a separate hidden subset and used for final ranking.

## Submission File

Your submission must be a CSV file named submission.csv with exactly two columns:

id: test example identifier

answer: predicted 0-indexed integer answer

Use this structure:

```csv
id,answer
test_02333,0
test_04102,2
test_00017,1
```

## Submission Rules

One prediction per test id.

id values must match the provided test file.

answer must be an integer index valid for that question’s choice set.

Files with missing columns, extra columns, invalid ids, or non-integer answers may be rejected or scored as invalid.

## Dataset Description

This competition uses a multimodal science question-answering dataset in which each example combines an image with textual context and multiple-choice options. Participants must predict the correct answer choice index for each question. The task requires both visual interpretation (e.g., diagrams, maps, charts) and language reasoning (question, hint, and lecture/context text).

The data is organized into three splits:

Train: labeled examples for model fitting.

Validation: labeled examples for model selection and tuning.

Test: unlabeled examples used for leaderboard scoring.

Each row corresponds to one question instance and includes:

id: unique example identifier.

image_path: relative path to the associated image file.

question: question text.

choices: JSON list of candidate answers.

num_choices: number of options in choices.

answer: 0-indexed correct option (available only in train/validation).

hint, lecture: optional textual context fields.

task, grade, subject, topic, category, skill: pedagogical metadata.

Images are stored by split under data/images/{train,val,test}.

A sample_submission.csv is provided with the required submission schema:

id

answer (0-indexed predicted choice)

The hidden test labels are withheld for evaluation, with leaderboard scoring performed on hidden ground truth.

## Local Data Layout

The competition data has been downloaded with `kagglehub` and exposed
through a project-local `data/` symlink layout. See `DATA.md`.

Observed split sizes:

- Train: 3109 rows / 3109 images.
- Validation: 1048 rows / 1048 images.
- Test: 1008 rows / 1008 images.

The CSV `image_path` values resolve relative to `data/`, e.g.
`images/train/train_07667.png` resolves to
`data/images/train/train_07667.png`.

## Starter Code And Execution Surface

The starter notebook at `src/starter_notebook.ipynb` is useful
reference code, but autonomous experiments should not depend on
Jupyter execution. Extract the data loading, prompt construction,
model loading, training, validation, and submission-writing logic into
plain `.py` files under `src/` or `scripts/` before running jobs.

The notebook identifies the baseline checkpoint as:

```text
HuggingFaceTB/SmolVLM-500M-Instruct
```

## Model And Cache Plan

The official base model should be downloaded and cached as an explicit
infrastructure preflight before any long training or evaluation run.
Use the exact checkpoint above, store the cache under `/scratch`, and
make offline execution explicit after the cache exists.

Expected preflight checks:

- download the checkpoint with `huggingface_hub.snapshot_download`;
- set `HF_HOME` / `TRANSFORMERS_CACHE` to a `/scratch` cache path;
- load `AutoProcessor` and `AutoModelForVision2Seq` once;
- verify adapter/freezing logic keeps trainable parameters at or
  below 5 million;
- record the offline env vars needed for later runs, such as
  `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.
