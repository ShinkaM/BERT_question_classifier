# %%
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import evaluate

# %%
import torch
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print(x)
else:
    print ("MPS device not found.")

# %%
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType

# Load your dataset from CSV files
def load_data(train_csv, test_csv):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # Convert labels to integers (0 or 1)
    train_df["is_question"] = train_df["is_question"].astype(int)

    # Convert the DataFrames to Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    return train_dataset, test_dataset




# %%
def preprocess_data(dataset, tokenizer, train = True):
    def tokenize_function(examples):
        return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=128)
    
    dataset = dataset.map(tokenize_function, batched=True)
    if train:
        dataset = dataset.rename_column("is_question", "labels")  # Rename label column to "labels" for Hugging Face compatibility
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    else:
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    return dataset

# %%
model_name = "cl-tohoku/bert-base-japanese"

# model_name =  "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model.to(device)
# Load and preprocess the dataset
train_dataset, test_dataset = load_data('datasets/train.csv', 'datasets/test.csv')
train_dataset = preprocess_data(train_dataset, tokenizer, True)
test_dataset = preprocess_data(test_dataset, tokenizer, False)

# %%
lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,  # Sequence classification task
        inference_mode=False,
        r=8,  # Rank of LoRA
        lora_alpha=32,  # Scaling factor
        lora_dropout=0.1,  # Dropout rate
    )

# %%
model = get_peft_model(model, lora_config)

# %%
training_args = TrainingArguments(
        output_dir="./results",          # Output directory
        evaluation_strategy="epoch",    # Evaluate every epoch
        learning_rate=2e-5,             # Learning rate
        per_device_train_batch_size=20, # Batch size for training
        per_device_eval_batch_size=20,  # Batch size for evaluation
        num_train_epochs=2,             # Number of epochs
        weight_decay=0.01,              # Weight decay
        logging_dir="./logs",           # Directory for logging
        logging_steps=10,               # Log every 10 steps
        save_strategy="epoch",          # Save model every epoch
        load_best_model_at_end=True,    # Load the best model at the end of training
        metric_for_best_model = 'eval_samples_per_second'
    )

# %%
def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,  # Optional: Define a function to compute metrics
    )

    # Train the model
trainer.train()

# %%



model.save_pretrained("./lora_bert_model")