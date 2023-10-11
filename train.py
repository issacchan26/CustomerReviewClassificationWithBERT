import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import evaluate
import numpy as np

def load_data(path, name):
    df = pd.read_csv(path)  
    df = df.rename(columns={'review': 'text', 'overall rating': 'label'})
    dataset = Dataset.from_pandas(df, split=name)
    return dataset

def preprocess_function(examples):
    return tokenizer(examples["text"], padding='max_length', truncation=True)

def compute_metrics(eval_pred):
    metric = evaluate.load('accuracy')
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

if __name__ == '__main__':

    # pd.set_option('display.max_rows', None)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained("LiYuan/amazon-review-sentiment-analysis")
    model = AutoModelForSequenceClassification.from_pretrained("LiYuan/amazon-review-sentiment-analysis")
    model = model.to(device)
    
    train_path = '/path to/data/train.csv'
    eval_path = '/path to/data/eval.csv'
    checkpoints_path = '/path to/checkpoints'

    train_dataset = load_data(train_path, name='train')
    eval_dataset = load_data(eval_path, name='eval')
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    eval_dataset = eval_dataset.map(preprocess_function, batched=True)

    training_args = TrainingArguments(
        output_dir=checkpoints_path,
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=20,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()





