"""Predict one new text with any of the saved homework classifiers."""
import argparse
import json
from pathlib import Path

import joblib
from experiment import average_embeddings, tokenize

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["binary_bow", "word_frequency", "tfidf", "glove", "word2vec_ag", "word2vec_nyt", "bert"], default="tfidf")
    parser.add_argument("--text", required=True)
    parser.add_argument("--models", type=Path, default=ROOT / "outputs/models")
    args = parser.parse_args()
    if args.method in {"binary_bow", "word_frequency", "tfidf"}:
        model = joblib.load(args.models / f"{args.method}.joblib")
        label = model.predict([args.text])[0]
    elif args.method == "bert":
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        torch.set_num_threads(4)
        model_path = args.models / "bert"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path).eval()
        encoded = tokenizer(args.text, max_length=64, truncation=True, padding="max_length", return_tensors="pt")
        with torch.inference_mode():
            idx = model(**encoded).logits.argmax(-1).item()
        label = model.config.id2label[idx]
    else:
        if args.method == "glove":
            vectors = joblib.load(args.models / "glove_subset.joblib")
        else:
            from gensim.models import Word2Vec
            vectors = Word2Vec.load(str(args.models / f"{args.method}.model")).wv
        matrix, _ = average_embeddings([tokenize(args.text)], vectors)
        label = joblib.load(args.models / f"{args.method}_classifier.joblib").predict(matrix)[0]
    print(json.dumps({"method": args.method, "label": str(label)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
