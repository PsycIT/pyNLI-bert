import random
import logging
from IPython.display import display, HTML

import numpy as np
import pandas as pd
import datasets
from datasets import load_dataset, load_metric, ClassLabel, Sequence
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification, EarlyStoppingCallback
import wandb
wandb.login()

# early stopping -> callback함수
# best metric 함게 해보는걸로 -> f1으로 해보는 (acc보단 정확 -> acc는 overfitting에 대응하지 못함)
# eval loss 오르고 있으므로 -> f1을 모니터링해서 early stopping -> ~~~~ 한쪽 클래스에 집중될수도~~~~

# model_checkpoint = "klue/roberta-base"
model_checkpoint = "klue/bert-base"
batch_size = 32
task = "nli"

from sklearn.model_selection import train_test_split

# HuggingFace datasets 라이브러리에 등록된 KLUE 데이터셋 중, NLI 다운로드
klue_dataset = load_dataset("klue", task)
from sklearn.model_selection import train_test_split

tr = klue_dataset['train']
test = klue_dataset['validation']

tr_dataset, eval_dataset = train_test_split(tr, test_size=0.2)

# 모델 성능 파악을 위한 메트릭 설정
# 기존 GLUE 데이터셋에 구현된 qnli의 accuracy 메트릭 load
metric = load_metric("glue", "qnli")

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

# cls_token에 해당하는 2번 토큰이 가장 좌측, sep_token의 3번 토큰이 중간과 가장 우측에 추가됨
sentence1_key, sentence2_key = ("premise", "hypothesis")
print(f"Sentence 1: {tr_dataset[sentence1_key][0]}")
print(f"Sentence 2: {tr_dataset[sentence2_key][0]}")

def preprocess_function(examples):
    return tokenizer(
        examples[sentence1_key],
        examples[sentence2_key],
        truncation=True,
        return_token_type_ids=False,
    )

preprocess_function(datasets["train"][:5])

encoded_datasets = datasets.map(preprocess_function, batched=True)

num_labels = 3
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


metric_name = "accuracy"

args = TrainingArguments(
    output_dir="nli-bert_with_early_stopping",
    evaluation_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=10,
    weight_decay=0.01,
    metric_for_best_model=metric_name,
    report_to="wandb",
    run_name="bert-nli-ep10-early-p2"
)

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_datasets["train"],
    eval_dataset=encoded_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()

trainer.evaluate()

wandb.finish()


# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#
