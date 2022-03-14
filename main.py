import random
import logging
from IPython.display import display, HTML

import numpy as np
import pandas as pd
import datasets
from datasets import load_dataset, load_metric, ClassLabel, Sequence
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification
import wandb
wandb.login()


# model_checkpoint = "klue/roberta-base"
model_checkpoint = "klue/bert-base"
batch_size = 32
task = "nli"

# HuggingFace datasets 라이브러리에 등록된 KLUE 데이터셋 중, NLI 다운로드
datasets = load_dataset("klue", task)
print(datasets)
print(datasets["train"][0])

# 데이터셋 확인 위한 시각화 함수
def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."

    picks = []

    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)

        # 이미 등록된 예제가 뽑힌 경우, 다시 추출
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)

        picks.append(pick)

    # 임의로 추출된 인덱스들로 구성된 데이터 프레임 선언
    df = pd.DataFrame(dataset[picks])

    for column, typ in dataset.features.items():
        # 라벨 클래스를 스트링으로 변환
        if isinstance(typ, ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])

    display(HTML(df.to_html()))

show_random_elements(datasets["train"])

# 모델 성능 파악을 위한 메트릭 설정
# 기존 GLUE 데이터셋에 구현된 qnli의 accuracy 메트릭 load
metric = load_metric("glue", "qnli")

fake_preds = np.random.randint(0, 2, size=(64,))
fake_labels = np.random.randint(0, 2, size=(64,))
metric.compute(predictions=fake_preds, references=fake_labels)



tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

# cls_token에 해당하는 2번 토큰이 가장 좌측, sep_token의 3번 토큰이 중간과 가장 우측에 추가됨
tmp = tokenizer("힛걸 진심 최고로 멋지다.", "힛걸 진심 최고다 그 어떤 히어로보다 멋지다")
print(tmp)

sentence1_key, sentence2_key = ("premise", "hypothesis")
print(f"Sentence 1: {datasets['train'][0][sentence1_key]}")
print(f"Sentence 2: {datasets['train'][0][sentence2_key]}")

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
    "nli-bert",
    evaluation_strategy="steps",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    report_to="wandb",
    run_name="bert-base-nli2"
)

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_datasets["train"],
    eval_dataset=encoded_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.evaluate()

wandb.finish()


# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#
