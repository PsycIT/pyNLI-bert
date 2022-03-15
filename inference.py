import datasets
from datasets import load_dataset, load_metric, ClassLabel, Sequence
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoConfig, \
    AutoModelForTokenClassification
from transformers import AutoModelForTokenClassification, AutoConfig, Trainer, DataCollatorForTokenClassification, TrainingArguments

import numpy as np
import os
import wandb
wandb.login()

model_checkpoint = "./nli-bert_with_early_stopping/checkpoint-2500/"
batch_size = 32
task = "nli"

klue_dataset = load_dataset("klue", task)
test_ds = klue_dataset['validation']

metric = load_metric("glue", "qnli")

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

sentence1_key, sentence2_key = ("premise", "hypothesis")
def preprocess_function(examples):
    encoding = tokenizer(
        examples[sentence1_key],
        examples[sentence2_key],
        truncation=True,
        return_token_type_ids=True,
        padding=True
    )
    encoding['labels'] = examples['label']
    return encoding

encoded_datasets_te = test_ds.map(preprocess_function, batched=True, remove_columns=test_ds.column_names)
config = AutoConfig.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, config=config)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


metric_name = "accuracy"

training_args = TrainingArguments(
    output_dir='./inference/',
)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

config = AutoConfig.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, config=config)
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=encoded_datasets_te,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)



result = trainer.evaluate()
print(result)