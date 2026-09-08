"""
PRACTICE 3 - EXERCISE 1: SENTIMENT ANALYSIS WITH HUGGING FACE
Mô tả: Sử dụng pre-trained sentiment analysis model từ Hugging Face Hub,
thực hiện phân tích cảm xúc câu và tìm hiểu quy trình Tokenization.
"""

from transformers import pipeline, AutoTokenizer

def run_exercise_1():
    print("=" * 60)
    print("EXERCISE 1: SENTIMENT ANALYSIS WITH HUGGING FACE")
    print("=" * 60)

    # 1. Load Pre-trained Sentiment Model với Pipeline
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    print(f"\n[1] Đang tải mô hình: {model_name}...")
    classifier = pipeline("sentiment-analysis", model=model_name)

    # 2. Chạy Sentiment Analysis trên các câu thử nghiệm
    test_sentences = [
        "I absolutely love learning Deep Learning with Hugging Face, it is fantastic!",
        "The service was terrible and the product arrived broken.",
        "The movie started at 8 PM yesterday in downtown.",
        "I don't dislike this product, but it could be better.",
        "The weather today is cloudy with a light breeze."
    ]

    print("\n[2] Kết quả phân tích cảm xúc trên 5 câu đại diện:")
    for idx, sentence in enumerate(test_sentences, 1):
        res = classifier(sentence)[0]
        print(f"  Câu {idx}: '{sentence}'")
        print(f"    => Sentiment: {res['label']} (Score: {res['score']:.4f})\n")

    # 3. Tìm hiểu quy trình Tokenization chi tiết
    print("[3] Phân tích quy trình Tokenization:")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sample_sentence = test_sentences[0]

    tokens = tokenizer.tokenize(sample_sentence)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    encoded = tokenizer(sample_sentence)

    print(f"  Sentence:       '{sample_sentence}'")
    print(f"  Tokens:         {tokens[:10]} ... (Tổng: {len(tokens)})")
    print(f"  Token IDs:      {token_ids[:10]} ...")
    print(f"  input_ids:      {encoded['input_ids'][:10]} ...")
    print(f"  attention_mask: {encoded['attention_mask'][:10]} ...")

if __name__ == "__main__":
    run_exercise_1()
