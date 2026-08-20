# PROMPT CHI TIẾT – PRACTICE 3: GET STARTED WITH HUGGING FACE

## Vai trò của AI

Bạn hãy đóng vai **giảng viên môn Deep Learning** và đồng thời là **trợ giảng lập trình Python**.  
Hãy hướng dẫn tôi hoàn thành bài tập **Practice 3 – Get started with Hugging Face** theo cách dễ hiểu, từng bước, phù hợp với sinh viên mới học Deep Learning.

## YÊU CẦU BẮT BUỘC VỀ JUPYTER NOTEBOOK

**Toàn bộ Practice 3 chỉ được thực hiện bằng Jupyter Notebook (`.ipynb`) theo yêu cầu của giảng viên.** Không tạo hoặc yêu cầu chạy file `.py`. Mọi phần cài thư viện, import, tải model, tokenization, dataset, fine-tuning, evaluation và test đều phải nằm trong các **Markdown Cell** và **Code Cell**.

Hãy đánh số rõ từng cell: `Cell 1`, `Cell 2`, `Cell 3`,... và cho biết đó là **Markdown Cell** hay **Code Cell**. Notebook phải có thể chạy tuần tự từ trên xuống dưới bằng **Restart Kernel and Run All**.

Trong notebook, ưu tiên cài thư viện bằng `%pip install ...`. Nếu vừa cài/cập nhật thư viện, hãy nhắc khi nào cần **Restart Kernel**.

Tôi muốn bạn **không chỉ đưa code**, mà phải giải thích rõ:
- Mục đích của từng bước.
- Ý nghĩa của từng thư viện.
- Ý nghĩa của từng đoạn code.
- Input là gì.
- Output dự kiến là gì.
- Những lỗi thường gặp và cách sửa.
- Những khái niệm Deep Learning liên quan.
- Cách trình bày kết quả để nộp bài.

Ngôn ngữ giải thích: **Tiếng Việt**.  
Tên hàm, tên thư viện, code và thuật ngữ kỹ thuật giữ nguyên bằng tiếng Anh khi cần thiết.

---

# BÀI TẬP CẦN THỰC HIỆN

## Practice 3 – Get started with Hugging Face

Bài tập gồm 2 phần.

---

# EXERCISE 1 – SENTIMENT ANALYSIS WITH HUGGING FACE

Yêu cầu:

1. Install the Hugging Face `transformers` library.
2. Use a pre-trained sentiment analysis model from the Hugging Face Hub.
3. Tokenize a sample sentence.
4. Perform sentiment analysis on the sentence.

## Mục tiêu Exercise 1

Hãy giúp tôi hiểu và thực hiện được toàn bộ quy trình:

```text
Input sentence
      ↓
Tokenizer
      ↓
Token IDs
      ↓
Pre-trained Model
      ↓
Prediction
      ↓
POSITIVE / NEGATIVE
```

---

## Yêu cầu hướng dẫn Exercise 1

### Bước 1 – Chuẩn bị môi trường

Hướng dẫn tôi cách kiểm tra:

```bash
python --version
pip --version
```

Sau đó hướng dẫn cài đặt các thư viện cần thiết.

Ví dụ:

```python
%pip install transformers torch
```

Chỉ hướng dẫn bằng **Jupyter Notebook**: cách tạo notebook, thêm Markdown Cell/Code Cell, chạy cell bằng `Shift + Enter`, Restart Kernel và `Run All` trước khi nộp.

Nếu có sự khác biệt giữa:

```bash
pip
pip3
python -m pip
```

thì giải thích ngắn gọn cho tôi.

---

### Bước 2 – Import thư viện

Hướng dẫn import các thư viện cần thiết.

Ví dụ:

```python
from transformers import pipeline
```

Giải thích:

- `transformers` là gì?
- `pipeline` là gì?
- Tại sao `pipeline` giúp sử dụng model dễ hơn?
- Hugging Face Hub là gì?

---

### Bước 3 – Load pre-trained sentiment model

Sử dụng một model phù hợp để phân tích cảm xúc tiếng Anh.

Ưu tiên model:

```text
distilbert-base-uncased-finetuned-sst-2-english
```

Ví dụ:

```python
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
```

Giải thích chi tiết:

- `classifier` là gì?
- `"sentiment-analysis"` nghĩa là gì?
- `model=` dùng để làm gì?
- Model được tải từ đâu?
- Model có phải train lại không?
- DistilBERT là gì?
- SST-2 là gì?

---

### Bước 4 – Test một câu

Ví dụ:

```python
sentence = "I love this movie!"
result = classifier(sentence)
print(result)
```

Giải thích output.

Ví dụ:

```text
[{'label': 'POSITIVE', 'score': 0.9998}]
```

Giải thích:

- `label`
- `POSITIVE`
- `NEGATIVE`
- `score`
- Score càng gần 1 có ý nghĩa gì?

Cho tôi thử ít nhất **5 câu**:

1. Một câu rất tích cực.
2. Một câu rất tiêu cực.
3. Một câu trung tính hoặc khó đoán.
4. Một câu có từ phủ định.
5. Một câu do bạn tự chọn.

Hiển thị kết quả dễ đọc.

---

### Bước 5 – Tokenization

Không được bỏ qua bước này.

Hướng dẫn tôi sử dụng:

```python
from transformers import AutoTokenizer

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(model_name)
```

Sau đó tokenize:

```python
sentence = "I love this movie!"

encoded = tokenizer(sentence)

print(encoded)
```

Giải thích rõ:

- Token là gì?
- Tokenization là gì?
- `AutoTokenizer` là gì?
- `from_pretrained()` làm gì?
- `input_ids` là gì?
- `attention_mask` là gì?

Sau đó cho tôi xem token thật:

```python
tokens = tokenizer.tokenize(sentence)
print(tokens)
```

Và token ID:

```python
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(token_ids)
```

Giải thích sự khác nhau giữa:

```text
Text
Token
Token ID
input_ids
attention_mask
```

---

### Bước 6 – Tổng kết Exercise 1

Sau khi hoàn thành, hãy viết cho tôi phần kết luận ngắn khoảng **5–7 câu** để đưa vào báo cáo.

Nội dung cần đề cập:

- Hugging Face.
- Transformers.
- Pre-trained model.
- Tokenizer.
- Sentiment Analysis.
- Kết quả thu được.

---

# EXERCISE 2 – FINETUNING A PRETRAINED MODEL FOR BINARY TEXT CLASSIFICATION

Yêu cầu bài tập:

1. Install necessary Hugging Face libraries:
   - `transformers`
   - `datasets`
   - `evaluate`

2. Load a simple dataset for binary text classification.
3. Load a pretrained model and tokenizer.
4. Preprocess the dataset.
5. Define training arguments.
6. Create a Trainer object and finetune the model.
7. Evaluate the finetuned model.

---

# Mục tiêu Exercise 2

Giúp tôi hiểu quy trình:

```text
Dataset
   ↓
Preprocessing
   ↓
Tokenizer
   ↓
Pre-trained Model
   ↓
Fine-tuning
   ↓
Evaluation
   ↓
Accuracy / F1 / Precision / Recall
```

---

## Bước 1 – Cài thư viện

Hướng dẫn cài:

```python
%pip install transformers datasets evaluate accelerate torch scikit-learn
```

Giải thích ngắn gọn chức năng từng thư viện:

```text
transformers
datasets
evaluate
accelerate
torch
scikit-learn
```

Nếu phiên bản thư viện có thể gây lỗi với `Trainer`, hãy hướng dẫn cách kiểm tra:

```bash
pip show transformers
pip show datasets
pip show accelerate
```

---

## Bước 2 – Load Dataset

Ưu tiên sử dụng dataset đơn giản phù hợp với binary sentiment classification.

Có thể sử dụng:

```python
from datasets import load_dataset

dataset = load_dataset("imdb")
```

Nếu IMDb quá lớn hoặc train quá lâu, hãy hướng dẫn lấy một tập nhỏ để học thử.

Ví dụ:

```python
small_train_dataset = dataset["train"].shuffle(seed=42).select(range(2000))
small_test_dataset = dataset["test"].shuffle(seed=42).select(range(500))
```

Giải thích:

- Dataset là gì?
- Train set là gì?
- Test set là gì?
- Shuffle là gì?
- `seed=42` để làm gì?
- `select()` để làm gì?
- Label `0` và `1` có ý nghĩa gì?

In thử một sample:

```python
print(dataset["train"][0])
```

Giải thích cấu trúc sample.

---

## Bước 3 – Load Tokenizer

Sử dụng:

```python
from transformers import AutoTokenizer

model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
```

Giải thích tại sao Exercise 2 có thể dùng:

```text
distilbert-base-uncased
```

thay vì model đã fine-tune sentiment.

---

## Bước 4 – Preprocess Dataset

Viết hàm:

```python
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )
```

Sau đó:

```python
tokenized_train = small_train_dataset.map(
    tokenize_function,
    batched=True
)

tokenized_test = small_test_dataset.map(
    tokenize_function,
    batched=True
)
```

Giải thích từng tham số:

```text
padding
truncation
max_length
batched
```

Giải thích tại sao model không thể nhận raw text trực tiếp.

---

# Bước 5 – Load Pretrained Model

Sử dụng:

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)
```

Giải thích:

- `AutoModelForSequenceClassification` là gì?
- Sequence Classification là gì?
- `num_labels=2` nghĩa là gì?
- Vì sao có cảnh báo một số weights được khởi tạo mới?
- Classification head là gì?

---

# Bước 6 – Define Training Arguments

Hướng dẫn sử dụng `TrainingArguments`.

Ví dụ phù hợp với máy cá nhân:

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=50,
    save_strategy="epoch"
)
```

Nếu phiên bản `transformers` hiện tại hỗ trợ tham số đánh giá tương ứng, hãy hướng dẫn cách bật evaluation trong quá trình train.

Giải thích:

- `output_dir`
- `num_train_epochs`
- `per_device_train_batch_size`
- `per_device_eval_batch_size`
- `learning_rate`
- `weight_decay`
- `logging_steps`
- `save_strategy`

Đặc biệt giải thích thật dễ hiểu:

### Epoch

```text
1 epoch = model đi qua toàn bộ training set một lần
```

### Batch size

```text
batch_size = số sample được đưa qua model trong một lần xử lý
```

### Learning rate

Giải thích learning rate ảnh hưởng đến quá trình học như thế nào.

---

# Bước 7 – Metrics

Tôi cần đánh giá model.

Hãy hướng dẫn tính:

```text
Accuracy
Precision
Recall
F1-score
```

Có thể sử dụng `evaluate` hoặc `sklearn.metrics`.

Ví dụ:

```python
import numpy as np
import evaluate

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return accuracy.compute(
        predictions=predictions,
        references=labels
    )
```

Nếu muốn tính thêm Precision, Recall và F1 thì viết code đầy đủ.

Giải thích ý nghĩa từng metric bằng ví dụ đơn giản.

---

# Bước 8 – Create Trainer

Viết:

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics
)
```

Giải thích:

- `Trainer` là gì?
- `model`
- `args`
- `train_dataset`
- `eval_dataset`
- `compute_metrics`

---

# Bước 9 – Fine-tuning

Chạy:

```python
trainer.train()
```

Giải thích trong quá trình này model đang làm gì.

Mô tả ngắn quy trình:

```text
Forward Pass
     ↓
Prediction
     ↓
Calculate Loss
     ↓
Backpropagation
     ↓
Update Weights
```

Giải thích:

- Forward propagation.
- Loss.
- Backpropagation.
- Optimizer.
- Update weights.

Không cần toán quá phức tạp nhưng phải giúp sinh viên mới học hiểu được.

---

# Bước 10 – Evaluate Model

Chạy:

```python
results = trainer.evaluate()

print(results)
```

Giải thích các giá trị có thể xuất hiện:

```text
eval_loss
eval_accuracy
eval_precision
eval_recall
eval_f1
eval_runtime
```

Giải thích cách nhận xét model dựa trên kết quả.

Ví dụ:

```text
Accuracy = 0.88
```

có thể hiểu rằng:

```text
Model dự đoán đúng khoảng 88% số sample trong tập evaluation.
```

---

# Bước 11 – Test Model Sau Khi Fine-tune

Sau khi train xong, hãy hướng dẫn test model bằng một câu mới.

Ví dụ:

```text
"This movie was absolutely amazing!"
```

và:

```text
"I regret watching this terrible movie."
```

Viết code đầy đủ để:

1. Tokenize câu.
2. Đưa vào model.
3. Lấy logits.
4. Chọn label có xác suất lớn nhất.
5. In ra:

```text
POSITIVE
```

hoặc:

```text
NEGATIVE
```

---

# Bước 12 – So sánh Exercise 1 và Exercise 2

Tạo bảng so sánh:

| Nội dung | Exercise 1 | Exercise 2 |
|---|---|---|
| Model | Pre-trained sentiment model | Pre-trained base model |
| Training | Không | Có |
| Dataset | Không bắt buộc | Có |
| Tokenizer | Có | Có |
| Fine-tuning | Không | Có |
| Evaluation | Đơn giản | Accuracy/F1/... |
| Độ khó | Cơ bản | Nâng cao |

Giải thích rõ:

```text
Exercise 1 = sử dụng model có sẵn.

Exercise 2 = lấy model có sẵn và huấn luyện thêm trên dataset của mình.
```

---

# YÊU CẦU VỀ JUPYTER NOTEBOOK

Chỉ tạo **một notebook hoàn chỉnh** có tên gợi ý:

```text
Practice3_HuggingFace.ipynb
```

Không cung cấp phiên bản `practice3.py`. Mỗi bước phải trình bày theo mẫu:

### Cell X – Markdown Cell
Dùng cho tiêu đề, mục tiêu, lý thuyết, giải thích và nhận xét.

### Cell X – Code Cell
Dùng cho Python code hoặc `%pip install`.

### Output dự kiến
Giải thích kết quả cần xuất hiện ngay dưới cell.

Notebook cần có thứ tự tối thiểu:

```text
Tiêu đề Practice 3
→ Chuẩn bị môi trường
→ Exercise 1: Sentiment Analysis
→ Tokenization
→ Test nhiều câu
→ Nhận xét Exercise 1
→ Exercise 2: Load Dataset
→ Preprocessing
→ Load Model + Tokenizer
→ TrainingArguments
→ Metrics
→ Trainer
→ Fine-tuning
→ Evaluation
→ Test câu mới
→ So sánh Exercise 1 và Exercise 2
→ Conclusion
```

Code phải có comment vừa đủ, dễ đọc, không dư thừa và phù hợp với sinh viên mới học. Notebook phải chạy được từ đầu đến cuối bằng **Restart Kernel and Run All** mà không phụ thuộc vào việc chạy cell sai thứ tự.

# YÊU CẦU VỀ GPU / CPU

Hãy kiểm tra:

```python
import torch

print(torch.cuda.is_available())
```

Nếu có NVIDIA GPU:

```text
CUDA available = True
```

thì giải thích cách model sử dụng GPU.

Nếu:

```text
False
```

thì hướng dẫn chạy bằng CPU.

Không bắt buộc phải có GPU mới làm được bài.

Nếu train IMDb quá lâu trên CPU, hãy tự động đề xuất giảm dataset xuống khoảng:

```text
500–2000 training samples
200–500 testing samples
```

để tôi có thể thực hành.

---

# CÁC LỖI CẦN HƯỚNG DẪN KHẮC PHỤC

Hãy thêm phần:

## Troubleshooting

Giải thích cách xử lý các lỗi phổ biến như:

### ModuleNotFoundError

Ví dụ:

```text
ModuleNotFoundError: No module named 'transformers'
```

Cách sửa.

---

### PyTorch chưa được cài

```text
No module named 'torch'
```

---

### Accelerate error

```text
Using the Trainer with PyTorch requires accelerate
```

Cách sửa:

```python
%pip install -U accelerate
```

---

### CUDA Out of Memory

Nếu gặp:

```text
CUDA out of memory
```

hãy hướng dẫn giảm:

```text
batch_size
max_length
dataset size
```

---

### Internet / Hugging Face download error

Nếu model hoặc dataset không tải được, hãy giải thích nguyên nhân có thể do:

- Internet.
- Firewall.
- Proxy.
- Hugging Face server.
- Cache.

---

### Parameter error trong TrainingArguments

Nếu một tham số đã thay đổi giữa các phiên bản `transformers`, hãy:

1. Giải thích nguyên nhân.
2. Kiểm tra phiên bản.
3. Đưa code tương thích với phiên bản hiện tại.

Không được đoán API nếu có khả năng đã thay đổi.

---

# YÊU CẦU PHẦN BÁO CÁO

Sau khi hoàn thành code, hãy giúp tôi viết nội dung báo cáo theo bố cục:

# PRACTICE 3 – GET STARTED WITH HUGGING FACE

## 1. Introduction

Giới thiệu ngắn về:

- Deep Learning.
- NLP.
- Hugging Face.
- Transformers.
- Pre-trained Model.
- Fine-tuning.

---

## 2. Exercise 1 – Sentiment Analysis

Bao gồm:

### 2.1 Objective

### 2.2 Libraries

### 2.3 Pre-trained Model

### 2.4 Tokenization

### 2.5 Sentiment Prediction

### 2.6 Results

### 2.7 Comments

---

## 3. Exercise 2 – Fine-tuning

Bao gồm:

### 3.1 Objective

### 3.2 Dataset

### 3.3 Preprocessing

### 3.4 Model and Tokenizer

### 3.5 Training Configuration

### 3.6 Fine-tuning

### 3.7 Evaluation

### 3.8 Testing

### 3.9 Results

---

## 4. Comparison

So sánh:

```text
Pre-trained Model
vs
Fine-tuned Model
```

---

## 5. Conclusion

Viết kết luận khoảng **1–2 đoạn**, ngắn gọn, đúng kiến thức sinh viên.

---

# YÊU CẦU ẢNH CHỤP KẾT QUẢ

Hãy chỉ rõ ở mỗi bước tôi nên chụp màn hình gì để đưa vào báo cáo.

Ví dụ:

```text
Hình 1 – Kết quả cài đặt transformers.

Hình 2 – Kết quả chạy sentiment analysis.

Hình 3 – Token và Token ID.

Hình 4 – Cấu trúc dataset.

Hình 5 – Quá trình training.

Hình 6 – Kết quả evaluation.

Hình 7 – Kết quả dự đoán câu mới.
```

Với mỗi hình, hãy đề xuất caption phù hợp.

---

# CÂU HỎI VẤN ĐÁP

Sau khi hoàn thành bài, hãy tạo cho tôi **15 câu hỏi vấn đáp** mà giảng viên có thể hỏi.

Phải bao gồm các câu hỏi về:

1. Hugging Face là gì?
2. Transformer là gì?
3. Pre-trained model là gì?
4. Fine-tuning là gì?
5. Tokenizer là gì?
6. Token là gì?
7. Token ID là gì?
8. Attention mask là gì?
9. Epoch là gì?
10. Batch size là gì?
11. Learning rate là gì?
12. Accuracy là gì?
13. F1-score là gì?
14. Exercise 1 khác Exercise 2 ở đâu?
15. Vì sao fine-tuning nhanh hơn train model từ đầu?

Mỗi câu hỏi phải có **đáp án mẫu ngắn gọn, dễ học thuộc**.

---

# PHONG CÁCH GIẢI THÍCH

Vì tôi là sinh viên mới học Deep Learning, hãy giải thích theo nguyên tắc:

```text
Khái niệm
↓
Giải thích đơn giản
↓
Ví dụ
↓
Code
↓
Output
↓
Ý nghĩa
```

Không giải thích quá học thuật ngay từ đầu.

Nếu gặp thuật ngữ khó như:

```text
logits
embedding
attention
backpropagation
optimizer
gradient
classification head
```

hãy giải thích theo cách đơn giản trước, sau đó mới nói ý nghĩa kỹ thuật.

---

# ĐỊNH DẠNG CÂU TRẢ LỜI MONG MUỐN

Hãy trả lời lần lượt:

1. Phân tích đề bài.
2. Chuẩn bị môi trường.
3. Exercise 1 từng bước.
4. Exercise 1 theo từng Markdown Cell và Code Cell.
5. Exercise 2 theo từng Markdown Cell và Code Cell.
6. Giải thích output và nhận xét của từng phần.
7. Giải thích kết quả.
8. Các lỗi thường gặp.
9. Nội dung báo cáo.
10. Danh sách ảnh cần chụp.
11. Kết luận.
12. 15 câu hỏi vấn đáp + đáp án.

Không bỏ qua bất kỳ bước nào.

Nếu code có thể thay đổi do phiên bản mới của Hugging Face/Transformers, hãy ưu tiên API hiện tại và ghi chú nếu khác với các phiên bản cũ.

Mục tiêu cuối cùng là sau khi làm theo hướng dẫn, tôi phải:

- Chạy được Exercise 1.
- Hiểu Tokenization.
- Chạy được Exercise 2.
- Fine-tune được model.
- Đánh giá được model.
- Hiểu các chỉ số Accuracy/F1.
- Có đủ hình để làm báo cáo.
- Có thể giải thích code khi giảng viên hỏi.
- Có file `Practice3_HuggingFace.ipynb` hoàn chỉnh để nộp.
- Không tạo file `.py`.
- Trước khi nộp phải thử `Restart Kernel and Run All` và bảo đảm không còn traceback/error.
