"""Homework 1: seven text classifiers on one persisted NYT split.

Run ``python experiment.py --help`` for individual stages.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import time
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Persist importable callable names even when invoked as a script.
if __name__ == "__main__":
    sys.modules["experiment"] = sys.modules[__name__]

ROOT = Path(__file__).resolve().parent
METHODS = ("binary_bow", "word_frequency", "tfidf", "glove", "word2vec_ag", "word2vec_nyt", "bert")
TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?")


def tokenize(text):
    """One shared English tokenizer for all classical methods."""
    return TOKEN_PATTERN.findall(str(text).lower())


tokenize.__module__ = "experiment"


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def environment():
    versions = {}
    for package in ["numpy", "scipy", "pandas", "scikit-learn", "gensim", "torch", "transformers", "reportlab"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions}


def read_data(path, labeled=True):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing data: {path}. Supply the complete course CSV.")
    if path.suffix.lower() != ".csv":
        raise ValueError("Only complete .csv files are accepted; do not use .wxdownload files.")
    # csv strict mode catches truncated quoted records before pandas parsing.
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        columns = reader.fieldnames or []
        if "text" not in columns or (labeled and "label" not in columns):
            raise ValueError(f"Required columns: text{' and label' if labeled else ''}; got {columns}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("CSV contains incomplete or extra fields.")
    df = pd.DataFrame(rows)
    if len(df) == 0 or df["text"].str.strip().eq("").any():
        raise ValueError("CSV is empty or contains blank texts.")
    if labeled and df["label"].str.strip().eq("").any():
        raise ValueError("Missing labels are not allowed.")
    # Original row IDs are zero-based positions after the header, not line numbers.
    df.insert(0, "row_id", np.arange(len(df)))
    return df


def prepare_data(args):
    df = read_data(args.nyt)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    sha = file_sha256(args.nyt)
    classes = sorted(df["label"].unique().tolist())
    if len(classes) < 2:
        raise ValueError("At least two classes are required.")
    permutation = np.random.default_rng(args.seed).permutation(len(df))
    # Shuffle once, then allocate floor(80% N), floor(10% N), remainder.
    n_train, n_val = int(0.8 * len(df)), int(0.1 * len(df))
    groups = {"train": permutation[:n_train].tolist(),
              "validation": permutation[n_train:n_train+n_val].tolist(),
              "test": permutation[n_train+n_val:].tolist()}
    if min(map(len, groups.values())) == 0:
        raise ValueError("Dataset is too small for an 80/10/10 split.")
    if set(df.iloc[groups["train"]]["label"]) != set(classes):
        raise ValueError("A label is absent in the training split. Use a different explicit seed.")
    identity = {"data_sha256": sha, "seed": args.seed, "indices": groups}
    split_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    split_path = out / "split.json"
    if split_path.exists() and json.loads(split_path.read_text(encoding="utf-8"))["split_hash"] != split_hash:
        raise ValueError("This output directory contains a different dataset/split. Choose a new --output directory.")
    parts = {name: df.iloc[ids].copy() for name, ids in groups.items()}
    counts = {name: part["label"].value_counts().reindex(classes, fill_value=0).to_dict() for name, part in parts.items()}
    text_sets = {name: set(part["text"].str.strip()) for name, part in parts.items()}
    overlap = {f"{a}_{b}": len(text_sets[a] & text_sets[b]) for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]}
    metadata = {**identity, "split_hash": split_hash, "classes": classes, "sizes": {k: len(v) for k, v in parts.items()},
                "class_counts": counts, "source_path": str(Path(args.nyt).resolve()),
                "duplicate_texts": int(df["text"].str.strip().duplicated().sum()), "cross_split_duplicate_texts": overlap,
                "tokenizer": TOKEN_PATTERN.pattern, "environment": environment()}
    save_json(split_path, metadata)
    records = [{"row_id": idx, "split": name} for name, ids in groups.items() for idx in ids]
    pd.DataFrame(records).sort_values("row_id").to_csv(out / "split_assignments.csv", index=False)
    print(f"NYT: {len(df)} rows; split={metadata['sizes']}; labels={classes}", flush=True)
    if any(overlap.values()):
        print(f"Source contains duplicate texts across splits: {overlap}. See report limitation.", flush=True)
    return parts, metadata


def metrics(y_true, y_pred, classes):
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0))}


def record_result(args, meta, name, y_pred, validation, config, elapsed, extra=None):
    test = read_data(args.nyt).iloc[meta["indices"]["test"]]
    truth = test["label"].tolist()
    result = {"method": name, "split_hash": meta["split_hash"], "data_sha256": meta["data_sha256"],
              "seed": args.seed, "validation": validation, "test": metrics(truth, y_pred, meta["classes"]),
              "config": config, "elapsed_seconds": elapsed,
              "classification_report": classification_report(truth, y_pred, labels=meta["classes"], output_dict=True, zero_division=0),
              "confusion_matrix": confusion_matrix(truth, y_pred, labels=meta["classes"]).tolist(),
              "classes": meta["classes"], "environment": environment(), **(extra or {})}
    folder = Path(args.output) / "metrics"
    save_json(folder / f"{name}.json", result)
    predictions = pd.DataFrame({"row_id": test["row_id"], "true_label": truth, "predicted_label": y_pred})
    predictions.to_csv(folder / f"{name}_predictions.csv", index=False)
    aggregate_results(args.output, meta["split_hash"])
    print(f"{name}: validation={validation}; test={result['test']}; {elapsed:.1f}s", flush=True)
    return result


def aggregate_results(output, split_hash):
    rows = []
    for name in METHODS:
        p = Path(output) / "metrics" / f"{name}.json"
        if p.exists():
            result = json.loads(p.read_text(encoding="utf-8"))
            if result["split_hash"] != split_hash:
                raise ValueError(f"Stale metrics with another split: {p}")
            rows.append({"method": name, "validation_accuracy": result["validation"]["accuracy"],
                         "validation_macro_f1": result["validation"]["macro_f1"],
                         "test_accuracy": result["test"]["accuracy"], "test_macro_f1": result["test"]["macro_f1"],
                         "elapsed_seconds": result["elapsed_seconds"]})
    pd.DataFrame(rows).to_csv(Path(output) / "results.csv", index=False)


def fit_logistic(x_train, y_train, x_val, y_val, args, classes, scale=False):
    best_model, best_c, best_metrics = None, None, None
    trials = []
    for c in args.c_values:
        lr = LogisticRegression(C=c, max_iter=2000, solver="lbfgs", random_state=args.seed)
        model = make_pipeline(StandardScaler(), lr) if scale else lr
        model.fit(x_train, y_train)
        score = metrics(y_val, model.predict(x_val), classes)
        trials.append({"C": c, **score})
        if best_metrics is None or (score["macro_f1"], score["accuracy"]) > (best_metrics["macro_f1"], best_metrics["accuracy"]):
            best_model, best_c, best_metrics = model, c, score
    return best_model, best_c, best_metrics, trials


def run_bow(args, parts, meta):
    for name in METHODS[:3]:
        started = time.monotonic()
        common = dict(tokenizer=tokenize, token_pattern=None, lowercase=False,
                      min_df=1, max_features=args.max_features, dtype=np.float64)
        vectorizer = TfidfVectorizer(**common) if name == "tfidf" else CountVectorizer(binary=name == "binary_bow", **common)
        x_train = vectorizer.fit_transform(parts["train"]["text"])
        x_val = vectorizer.transform(parts["validation"]["text"])
        classifier, c, val, trials = fit_logistic(x_train, parts["train"]["label"], x_val, parts["validation"]["label"], args, meta["classes"])
        # Transform/evaluate test only after choosing C on validation.
        pred = classifier.predict(vectorizer.transform(parts["test"]["text"]))
        model_path = Path(args.output) / "models" / f"{name}.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(make_pipeline(vectorizer, classifier), model_path)
        config = {"C": c, "C_candidates": args.c_values, "vocabulary_size": len(vectorizer.vocabulary_),
                  "max_features": args.max_features, "min_df": 1, "solver": "lbfgs", "max_iter": 2000,
                  "class_weight": None, "tokenizer": TOKEN_PATTERN.pattern}
        record_result(args, meta, name, pred, val, config, time.monotonic()-started, {"validation_trials": trials})


def load_glove(path, needed):
    vectors = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            word, _, values = line.rstrip().partition(" ")
            if word in needed:
                vector = np.fromstring(values, sep=" ", dtype=np.float32)
                if vector.shape != (100,) or not np.isfinite(vector).all():
                    raise ValueError(f"Invalid 100-dimensional GloVe vector for {word}")
                vectors[word] = vector
    if not vectors:
        raise ValueError("GloVe contains no matching words; check the file and tokenizer.")
    return vectors


def average_embeddings(documents, vectors, dimension=100):
    matrix = np.zeros((len(documents), dimension), dtype=np.float32)
    valid_count = total_count = zero_docs = 0
    for i, tokens in enumerate(documents):
        found = [vectors[word] for word in tokens if word in vectors]
        total_count += len(tokens)
        valid_count += len(found)
        if found:
            matrix[i] = np.mean(found, axis=0)
        else:
            zero_docs += 1
    return matrix, {"token_coverage": valid_count / total_count if total_count else 0.0,
                    "valid_tokens": valid_count, "total_tokens": total_count, "zero_vector_documents": zero_docs}


def run_embeddings(args, parts, meta):
    from gensim.models import Word2Vec

    tokenized = {k: [tokenize(t) for t in part["text"]] for k, part in parts.items()}
    names = ["glove", "word2vec_ag", "word2vec_nyt"] if args.embedding == "all" else [args.embedding]
    for name in names:
        started = time.monotonic()
        model_folder = Path(args.output) / "models"
        model_folder.mkdir(parents=True, exist_ok=True)
        if name == "glove":
            if not Path(args.glove).exists():
                raise FileNotFoundError("Run python download_resources.py glove, or supply --glove.")
            # Fixed external vectors can be looked up for unseen evaluation words.
            # No NYT validation/test text is used to FIT an embedding.
            needed = {word for docs in tokenized.values() for tokens in docs for word in tokens}
            vectors = load_glove(args.glove, needed)
            joblib.dump(vectors, model_folder / "glove_subset.joblib")
            config = {"dimension": 100, "embedding": "glove.6B.100d", "source_sha256": file_sha256(args.glove),
                      "pooling": "mean of in-vocabulary token occurrences"}
        else:
            sentences = [tokenize(t) for t in read_data(args.ag, labeled=False)["text"]] if name == "word2vec_ag" else tokenized["train"]
            config = {"dimension": 100, "window": 5, "min_count": 2, "sg": 1, "negative": 5,
                      "epochs": 10, "workers": args.workers, "seed": args.seed,
                      "training_corpus": "AG News supplied CSV" if name == "word2vec_ag" else "NYT training split only",
                      "corpus_documents": len(sentences), "pooling": "mean of in-vocabulary token occurrences"}
            if name == "word2vec_ag":
                config["source_sha256"] = file_sha256(args.ag)
            model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=2, sg=1,
                             negative=5, epochs=10, workers=args.workers, seed=args.seed,
                             hashfxn=stable_hash)
            model.save(str(model_folder / f"{name}.model"))
            vectors = model.wv
        features, coverage = {}, {}
        for split, docs in tokenized.items():
            features[split], coverage[split] = average_embeddings(docs, vectors)
        classifier, c, val, trials = fit_logistic(features["train"], parts["train"]["label"], features["validation"], parts["validation"]["label"], args, meta["classes"], scale=True)
        prediction = classifier.predict(features["test"])
        joblib.dump(classifier, model_folder / f"{name}_classifier.joblib")
        config.update({"C": c, "C_candidates": args.c_values, "standard_scaler": "fit on NYT train only", "solver": "lbfgs", "max_iter": 2000})
        record_result(args, meta, name, prediction, val, config, time.monotonic()-started,
                      {"coverage": coverage, "validation_trials": trials, "embedding_vocabulary_size": len(vectors)})


def stable_hash(word):
    """Avoid Python's process-specific hash when initializing Word2Vec."""
    return int.from_bytes(hashlib.md5(word.encode("utf-8")).digest()[:4], "little")


stable_hash.__module__ = "experiment"


def run_bert(args, parts, meta):
    import math
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    started = time.monotonic()
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError("CUDA was requested but is unavailable.")
    classes = meta["classes"]
    label2id = {name: i for i, name in enumerate(classes)}
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    encoded = {}
    truncation = {}
    for split, part in parts.items():
        texts = part["text"].tolist()
        all_lengths = [len(tokenizer(t, truncation=False, verbose=False)["input_ids"]) for t in texts]
        truncation[split] = {"fraction_over_64": float(np.mean(np.array(all_lengths) > 64)),
                             "mean_original_wordpieces": float(np.mean(all_lengths))}
        encoded[split] = tokenizer(texts, max_length=64, truncation=True, padding="max_length", return_tensors="pt")

    class EncodedDataset(Dataset):
        def __init__(self, split):
            self.encodings = encoded[split]
            self.labels = torch.tensor([label2id[y] for y in parts[split]["label"]], dtype=torch.long)
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, i):
            return {**{k: v[i] for k, v in self.encodings.items()}, "labels": self.labels[i]}

    generator = torch.Generator().manual_seed(args.seed)
    loaders = {split: DataLoader(EncodedDataset(split), batch_size=args.batch_size if split == "train" else 32,
                                shuffle=split == "train", generator=generator if split == "train" else None,
                                num_workers=0, pin_memory=device.type == "cuda") for split in parts}
    model = AutoModelForSequenceClassification.from_pretrained(args.bert_model, num_labels=len(classes),
                id2label={i: name for i, name in enumerate(classes)}, label2id=label2id).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(loaders["train"]) / args.gradient_accumulation)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * steps_per_epoch * 3), num_training_steps=steps_per_epoch * 3)
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    model_folder = Path(args.output) / "models/bert"
    model_folder.mkdir(parents=True, exist_ok=True)

    def evaluate(split):
        model.eval()
        predictions = []
        with torch.inference_mode():
            for batch in loaders[split]:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    logits = model(**batch).logits
                predictions.extend(logits.argmax(-1).cpu().tolist())
        labels = [classes[i] for i in predictions]
        return labels, metrics(parts[split]["label"], labels, classes)

    print(f"BERT training on {device}; 3 epochs; max_length=64", flush=True)
    history, best_val, best_epoch = [], None, None
    for epoch in range(1, 4):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = examples = 0
        for step, batch in enumerate(loaders["train"], 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            # The final partial accumulation window uses its actual size.
            window_start = ((step - 1) // args.gradient_accumulation) * args.gradient_accumulation
            denominator = min(args.gradient_accumulation, len(loaders["train"]) - window_start)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                loss = model(**batch).loss
            loss_sum += float(loss.detach()) * len(batch["labels"])
            examples += len(batch["labels"])
            scaler.scale(loss / denominator).backward()
            if step % args.gradient_accumulation == 0 or step == len(loaders["train"]):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                # AMP may skip an optimizer update after a non-finite gradient.
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if step % 50 == 0:
                print(f"epoch {epoch}/3 batch {step}/{len(loaders['train'])} loss={loss_sum/examples:.4f}", flush=True)
        _, val = evaluate("validation")
        history.append({"epoch": epoch, "train_loss": loss_sum/examples, **val})
        save_json(Path(args.output) / "bert_history.json", history)
        print(f"epoch {epoch}: {history[-1]}", flush=True)
        if best_val is None or (val["macro_f1"], val["accuracy"]) > (best_val["macro_f1"], best_val["accuracy"]):
            best_val, best_epoch = val, epoch
            # Store only the best state; optimizer need not be kept for evaluation.
            torch.save(model.state_dict(), model_folder / "best_state.pt")
    model.load_state_dict(torch.load(model_folder / "best_state.pt", map_location=device, weights_only=True))
    predictions, _ = evaluate("test")
    model.save_pretrained(model_folder)
    tokenizer.save_pretrained(model_folder)
    # Avoid keeping two copies of the same 440 MB checkpoint.
    (model_folder / "best_state.pt").unlink()
    config = {"model": "google-bert/bert-base-uncased", "local_or_hub_path": str(args.bert_model),
              "max_length": 64, "epochs": 3, "selected_epoch": best_epoch, "batch_size": args.batch_size,
              "gradient_accumulation": args.gradient_accumulation, "learning_rate": 2e-5,
              "weight_decay": 0.01, "warmup_ratio": 0.1, "device": str(device), "fp16": amp_enabled,
              "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
              "seed": args.seed, "checkpoint_selection": "validation Macro-F1 then Accuracy"}
    source = Path(args.bert_model) / "source.json"
    if source.exists():
        config["pretrained_source"] = json.loads(source.read_text(encoding="utf-8"))
    record_result(args, meta, "bert", predictions, best_val, config, time.monotonic()-started,
                  {"history": history, "truncation": truncation})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["prepare", "bow", "embeddings", "bert", "all"], default="all")
    parser.add_argument("--nyt", type=Path, default=ROOT / "HW-1/nyt.csv")
    parser.add_argument("--ag", type=Path, default=ROOT / "HW-1/ag.csv")
    parser.add_argument("--glove", type=Path, default=ROOT / "data/glove.6B.100d.txt")
    parser.add_argument("--bert-model", default=str(ROOT / "models/bert-base-uncased"))
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c-values", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--embedding", choices=["all", "glove", "word2vec_ag", "word2vec_nyt"], default="all")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if min(args.c_values) <= 0 or min(args.batch_size, args.gradient_accumulation, args.workers, args.threads) < 1:
        parser.error("C values and counts must be positive.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    parts, meta = prepare_data(args)
    if args.task in {"bow", "all"}:
        run_bow(args, parts, meta)
    if args.task in {"embeddings", "all"}:
        run_embeddings(args, parts, meta)
    if args.task in {"bert", "all"}:
        run_bert(args, parts, meta)


if __name__ == "__main__":
    main()
