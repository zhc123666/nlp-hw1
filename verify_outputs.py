"""Reload saved models and check their predictions against the experiment logs."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from experiment import METHODS, ROOT, average_embeddings, file_sha256, read_data, save_json, tokenize


def main():
    import torch
    from gensim.models import Word2Vec
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(4)
    output = ROOT / "outputs"
    meta = json.loads((output / "split.json").read_text(encoding="utf-8"))
    assert file_sha256(ROOT / "HW-1/nyt.csv") == meta["data_sha256"]
    texts = read_data(ROOT / "HW-1/nyt.csv").iloc[meta["indices"]["test"]]["text"].tolist()
    tokens = [tokenize(text) for text in texts]
    statuses = {}
    for name in METHODS:
        if name in METHODS[:3]:
            prediction = joblib.load(output / "models" / f"{name}.joblib").predict(texts).tolist()
        elif name != "bert":
            vectors = joblib.load(output / "models/glove_subset.joblib") if name == "glove" else Word2Vec.load(str(output / "models" / f"{name}.model")).wv
            features, _ = average_embeddings(tokens, vectors)
            prediction = joblib.load(output / "models" / f"{name}_classifier.joblib").predict(features).tolist()
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            path = output / "models/bert"
            tokenizer = AutoTokenizer.from_pretrained(path)
            model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
            encoded = tokenizer(texts, max_length=64, truncation=True, padding="max_length", return_tensors="pt")
            assert encoded["input_ids"].shape[1] == 64
            prediction = []
            with torch.inference_mode():
                for start in range(0, len(texts), 32):
                    batch = {key: value[start:start+32].to(device) for key, value in encoded.items()}
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                        ids = model(**batch).logits.argmax(-1).cpu().tolist()
                    prediction.extend(model.config.id2label[i] for i in ids)
        recorded = pd.read_csv(output / "metrics" / f"{name}_predictions.csv")["predicted_label"].tolist()
        if prediction != recorded:
            raise AssertionError(f"Reloaded {name} predictions disagree with saved predictions.")
        statuses[name] = {"passed": True, "test_predictions_checked": len(prediction)}
        print(f"{name}: {len(prediction)} persisted predictions match", flush=True)
    save_json(output / "verification.json", {"split_hash": meta["split_hash"], "models": statuses})


if __name__ == "__main__":
    main()
