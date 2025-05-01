
# train.py
import os
import mlflow
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# ---- CONFIG SECTION ---- #
MLFLOW_TRACKING_URI = "http://<NODE1_IP>:8000"
MLFLOW_S3_ENDPOINT_URL = "http://<NODE1_IP>:9000"
AWS_ACCESS_KEY_ID = "your-access-key"
AWS_SECRET_ACCESS_KEY = "your-secret-key"
EXPERIMENT_NAME = "bert-training"

MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 2
LR = 2e-5
# ------------------------ #

# Set up MLflow
os.environ["MLFLOW_S3_ENDPOINT_URL"] = MLFLOW_S3_ENDPOINT_URL
os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# Load example dataset
dataset = load_dataset("imdb")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=MAX_LEN)

encoded = dataset.map(tokenize, batched=True)
encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])

model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

training_args = TrainingArguments(
    output_dir="./outputs",
    evaluation_strategy="epoch",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    logging_dir="./logs",
    logging_steps=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded["train"].select(range(2000)),
    eval_dataset=encoded["test"].select(range(500))
)

with mlflow.start_run():
    mlflow.log_params({
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "model": "bert-base-uncased"
    })

    trainer.train()

    torch.save(model.state_dict(), "outputs/bert_model.pt")
    mlflow.log_artifact("outputs/bert_model.pt")
