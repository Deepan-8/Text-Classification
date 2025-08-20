import streamlit as st
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import os

# Streamlit title
st.title("Custom Text Classification with Hugging Face Transformers")

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV (must contain 'data' and 'label' columns)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Dataset")
    st.dataframe(df.head())

    # Check for required columns
    required_cols = ["data", "label"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.stop()

    # Label encoding
    label_names = df["label"].unique().tolist()
    label2id = {label: i for i, label in enumerate(label_names)}
    id2label = {i: label for label, i in label2id.items()}
    df["label"] = df["label"].map(label2id)

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=0.2)

    # Load tokenizer
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize function
    def tokenize(batch):
        return tokenizer(batch["data"], padding="max_length", truncation=True)

    dataset = dataset.map(tokenize, batched=True)

    # Remove columns if they exist
    columns_to_remove = ["data", "__index_level_0__"]
    existing_columns = dataset["train"].column_names
    columns_to_remove = [col for col in columns_to_remove if col in existing_columns]
    dataset = dataset.remove_columns(columns_to_remove)
    dataset.set_format("torch")

    # Load model
    num_labels = len(label_names)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    # Evaluation metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="weighted"),
        }

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
    )


    # Train button
    if st.button("Train Model"):
        with st.spinner("Training in progress..."):
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset["train"],
                eval_dataset=dataset["test"],
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
            )
            trainer.train()

        st.success("✅ Training complete!")

        # Save model
        model.save_pretrained("./saved_model")
        tokenizer.save_pretrained("./saved_model")

# Inference section
if os.path.exists("./saved_model"):
    st.write("### Try Real-time Predictions")
    user_input = st.text_input("Enter a sentence:")
    if user_input:
        clf = pipeline(
            "text-classification",
            model="./saved_model",
            tokenizer="./saved_model"
        )
        prediction = clf(user_input)[0]
        st.write(
            f"**Prediction:** `{prediction['label']}`\n\n"
            f"**Confidence:** `{prediction['score']:.4f}`"
        )
