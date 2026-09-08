"""
PRACTICE 3 - EXERCISE 2: FINETUNING A PRETRAINED MODEL FOR BINARY TEXT CLASSIFICATION
Mô tả: Huấn luyện tinh chỉnh (Fine-tune) mô hình DistilBERT trên tập dữ liệu phân loại phân cảm xúc IMDb.
"""

import torch
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

def run_exercise_2():
    print("=" * 60)
    print("EXERCISE 2: FINE-TUNING PRETRAINED MODEL (IMDb DATASET)")
    print("=" * 60)

    # 1. Load Dataset IMDb và lấy subset nhỏ để thực hành nhanh
    print("\n[1] Đang tải IMDb dataset...")
    raw_dataset = load_dataset("stanfordnlp/imdb")

    print("    Tạo tập dữ liệu nhỏ (Small Subset)...")
    small_train_dataset = raw_dataset["train"].shuffle(seed=42).select(range(500))
    small_test_dataset = raw_dataset["test"].shuffle(seed=42).select(range(100))

    # 2. Load Tokenizer & Preprocess Dataset
    base_model_name = "distilbert-base-uncased"
    print(f"\n[2] Đang tải Tokenizer: {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    print("    Tiền xử lý (tokenize) tập Train và Test...")
    tokenized_train = small_train_dataset.map(tokenize_function, batched=True)
    tokenized_test = small_test_dataset.map(tokenize_function, batched=True)

    # 3. Load Base Model & Cấu hình TrainingArguments
    print(f"\n[3] Load Pretrained Model và cấu hình Trainer...")
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=2
    )

    eval_param_key = "eval_strategy" if hasattr(TrainingArguments("test"), "eval_strategy") else "evaluation_strategy"
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        **{eval_param_key: "epoch"},
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics
    )

    # 4. Thực hiện Fine-tuning
    print("\n[4] Bắt đầu quá trình Fine-tuning...")
    trainer.train()

    # 5. Đánh giá Model trên tập Test
    print("\n[5] Đánh giá mô hình trên tập Test...")
    eval_results = trainer.evaluate()
    print(f"    - Eval Loss:     {eval_results['eval_loss']:.4f}")
    print(f"    - Accuracy:      {eval_results['eval_accuracy']:.4f} ({eval_results['eval_accuracy']*100:.2f}%)")
    print(f"    - Precision:     {eval_results['eval_precision']:.4f}")
    print(f"    - Recall:        {eval_results['eval_recall']:.4f}")
    print(f"    - F1 Score:      {eval_results['eval_f1']:.4f}")

    # 6. Dự đoán mẫu thử nghiệm mới
    print("\n[6] Dự đoán các câu thử nghiệm mới:")
    new_sentences = [
        "This movie was absolutely amazing with brilliant acting!",
        "I regret watching this terrible movie, it was a waste of time."
    ]

    model.eval()
    for sentence in new_sentences:
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
        predicted_class_id = torch.argmax(probabilities).item()
        label_map = {0: "NEGATIVE", 1: "POSITIVE"}
        predicted_label = label_map[predicted_class_id]
        confidence = probabilities[predicted_class_id].item()

        print(f"  Câu: '{sentence}'")
        print(f"    => Sentiment: {predicted_label} (Confidence: {confidence:.4f})\n")

if __name__ == "__main__":
    run_exercise_2()
