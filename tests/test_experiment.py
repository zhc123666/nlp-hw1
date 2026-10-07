"""Check data isolation, representation semantics, and persisted inference."""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment import average_embeddings, metrics, prepare_data, read_data, tokenize


def test_binary_count_and_tfidf_no_test_vocabulary():
    train = ["apple apple banana", "banana pear"]
    binary = CountVectorizer(binary=True, tokenizer=tokenize, token_pattern=None).fit(train)
    counts = CountVectorizer(tokenizer=tokenize, token_pattern=None).fit(train)
    assert binary.transform([train[0]])[0, binary.vocabulary_["apple"]] == 1
    assert counts.transform([train[0]])[0, counts.vocabulary_["apple"]] == 2
    tfidf = TfidfVectorizer(tokenizer=tokenize, token_pattern=None).fit(train)
    old_idf = tfidf.idf_.copy()
    assert tfidf.transform(["unseen_test_word"]).nnz == 0
    assert "unseen" not in tfidf.vocabulary_
    np.testing.assert_array_equal(tfidf.idf_, old_idf)


def test_mean_pooling_occurrences_oov_and_empty():
    vectors = {"apple": np.ones(100), "banana": np.full(100, 4.0)}
    matrix, stats = average_embeddings([["apple", "apple", "banana", "oov"], ["oov"], []], vectors)
    np.testing.assert_allclose(matrix[0], 2.0)
    np.testing.assert_array_equal(matrix[1:], np.zeros((2, 100)))
    assert stats["token_coverage"] == 3/5
    assert stats["zero_vector_documents"] == 2
    assert np.isfinite(matrix).all()


def test_split_is_reproducible_disjoint_and_rejects_stale_data(tmp_path):
    path = tmp_path / "nyt.csv"
    pd.DataFrame({"text": [f"article {i}" for i in range(100)], "label": ["a", "b"]*50}).to_csv(path, index=False)
    args = argparse.Namespace(nyt=path, output=tmp_path / "out", seed=42)
    parts, first = prepare_data(args)
    _, second = prepare_data(args)
    assert first["split_hash"] == second["split_hash"]
    sets = [set(part["row_id"]) for part in parts.values()]
    assert list(map(len, sets)) == [80, 10, 10]
    assert sets[0].isdisjoint(sets[1]) and sets[0].isdisjoint(sets[2]) and sets[1].isdisjoint(sets[2])
    assert set.union(*sets) == set(range(100))
    data = pd.read_csv(path)
    data.loc[0, "text"] = "changed"
    data.to_csv(path, index=False)
    with pytest.raises(ValueError, match="different dataset"):
        prepare_data(args)


def test_truncated_download_and_missing_columns_rejected(tmp_path):
    path = tmp_path / "nyt.csv"
    path.write_text('text,label\n"unfinished,a', encoding="utf-8")
    with pytest.raises(Exception):
        read_data(path)
    path.write_text("text\na valid text\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Required columns"):
        read_data(path)


def test_macro_f1_includes_missing_prediction_classes():
    score = metrics(["a", "a", "b"], ["a", "a", "a"], ["a", "b"])
    assert score["accuracy"] == 2/3
    assert score["macro_f1"] == pytest.approx(0.4)


def test_saved_pipeline_has_importable_tokenizer(tmp_path):
    pipeline = make_pipeline(CountVectorizer(tokenizer=tokenize, token_pattern=None), LogisticRegression())
    pipeline.fit(["finance bank", "bank market", "soccer goal", "soccer team"], [0, 0, 1, 1])
    path = tmp_path / "model.joblib"
    joblib.dump(pipeline, path)
    restored = joblib.load(path)
    assert restored.predict(["bank", "soccer"]).tolist() == [0, 1]
