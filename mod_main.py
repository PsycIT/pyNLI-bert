import os
import numpy as np

import datasets
from datasets import load_dataset, load_metric, ClassLabel, Sequence
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoConfig, \
    AutoModelForTokenClassification
from transformers import AutoModelForSequenceClassification, EarlyStoppingCallback
import wandb
wandb.login()

# early stopping -> callback함수
# best metric 함게 해보는걸로 -> f1으로 해보는 (acc보단 정확 -> acc는 overfitting에 대응하지 못함)
# eval loss 오르고 있으므로 -> f1을 모니터링해서 early stopping -> ~~~~ 한쪽 클래스에 집중될수도~~~~

model_checkpoint = "klue/bert-base"
batch_size = 32
task = "nli"


# HuggingFace datasets 라이브러리에 등록된 KLUE 데이터셋 중, NLI 다운로드
klue_dataset = load_dataset("klue", task)
test_ds = klue_dataset['validation']

klue_dataset = klue_dataset['train'].train_test_split(test_size=0.1)
train_ds = klue_dataset['train']
eval_ds = klue_dataset['test']


# 모델 성능 파악을 위한 메트릭 설정
# 기존 GLUE 데이터셋에 구현된 qnli의 accuracy 메트릭 load
metric = load_metric("glue", "qnli")

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

# cls_token에 해당하는 2번 토큰이 가장 좌측, sep_token의 3번 토큰이 중간과 가장 우측에 추가됨
sentence1_key, sentence2_key = ("premise", "hypothesis")
print(f"Sentence 1: {train_ds[sentence1_key][0]}")
print(f"Sentence 2: {train_ds[sentence2_key][0]}")

def preprocess_function(examples):
    return tokenizer(
        examples[sentence1_key],
        examples[sentence2_key],
        truncation=True,
        return_token_type_ids=True,
    )

encoded_datasets_tr = train_ds.map(preprocess_function, batched=True)
encoded_datasets_eval = eval_ds.map(preprocess_function, batched=True)
encoded_datasets_te = test_ds.map(preprocess_function, batched=True)

num_labels = 3
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


metric_name = "accuracy"

args = TrainingArguments(
    output_dir="nli-bert_with_early_stopping2",
    evaluation_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=10,
    weight_decay=0.01,
    metric_for_best_model=metric_name,
    report_to="wandb"
#    run_name="bert-nli-ep10-early-p1-2"
)

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_datasets_tr,
    eval_dataset=encoded_datasets_eval,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
)

wandb.init(project='pyNLI-klue', entity='psycit', name='bert-nli-ep10-early-stopping_inference')
trainer.train()

trainer.evaluate()

#
# def inference(test_ds):
#     training_args = TrainingArguments(per_device_eval_batch_size=1, output_dir='./inference/')
#     os.environ["TOKENIZERS_PARALLELISM"] = "false"
#
#     model_path = './nli-bert_with_early_stopping2/checkpoint-2500/'
#     config = AutoConfig.from_pretrained(model_path)
#     model = AutoModelForTokenClassification.from_pretrained(model_path, config=config)
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=None,
#         eval_dataset=test_ds,
#         tokenizer=tokenizer,
#     )
#     predictions, _, _ = trainer.predict(test_dataset=test_ds)
#
#     return predictions
#
# pred = inference(encoded_datasets_te)
#
# compute_metrics(pred)

wandb.finish()


# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#
