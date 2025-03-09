import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import evaluate
import torch
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


"""
Load the dataset from pandas and convert boolean to int for easier computation

def load_data(train_csv):
    df = pd.read_csv(train_csv)
    
    df["is_question"] = df["is_question"].astype(int)
    #make train and val set, using 80% for training
    msk = np.random.rand(len(df)) < 0.8
    train = df[msk]
    val = df[~msk]
    train_dataset = Dataset.from_pandas(train)
    val_dataset = Dataset.from_pandas(val)
    return train_dataset, val_dataset

"""

# Preprocess data by tokenizing the sentences and relabeling the column.
# The tokenizer base for the model used "cl-tohoku/bert-base-japanese" is MeCab

"""

def preprocess_data(dataset, tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=128)
    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset.rename_column("is_question", "labels")  # Rename for Hugging Face compatibility
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset

"""
# Training code
# Execute the code below to train the model using LORA; 100iters/60sec on Mac M2

"""

model_name = "cl-tohoku/bert-base-japanese"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu") 
# device = torch.device("cuda:0" if torch.cuda.is_available()  else "cpu" )

model.to(device)

train_dataset, val_dataset = load_data('datasets/train.csv')
train_dataset = preprocess_data(train_dataset, tokenizer)
val_dataset = preprocess_data(val_dataset, tokenizer)

lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,  # Sequence classification task
        inference_mode=False,
        r=8,  # Rank of LoRA
        lora_alpha=32,  # Scaling factor
        lora_dropout=0.1,  # Dropout rate
    )

model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
        output_dir="./results",         
        evaluation_strategy="epoch",   
        learning_rate=2e-5,             
        per_device_train_batch_size=20, 
        per_device_eval_batch_size=20,  
        num_train_epochs=3,             
        weight_decay=0.01,              
        logging_dir="./logs",          
        logging_steps=10,              
        save_strategy="epoch",        
        load_best_model_at_end=True,   
        metric_for_best_model = 'eval_samples_per_second'
    )

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics, 
    )

trainer.train()

model.save_pretrained("./lora_bert_model_v2")

"""

"""
Load fine-tuned model

"""
MODEL_NAME = "./lora_bert_model_v2"#path to finetuned model
model_name = "cl-tohoku/bert-base-japanese"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

def predict_intent(text, model, tokenizer):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128,
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()
    return predicted_class

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu") ##if mps is available
# device = torch.device("cuda:0" if torch.cuda.is_available()  else "cpu" )
model.to(device)

df = pd.read_csv('datasets/test.csv')

results = []        
for t in df['sentence']:
    results.append(predict_intent(t, model, tokenizer))

results = [True if i else False for i in results]

df['is_question'] = results

df.to_csv('output.csv', index=None)
