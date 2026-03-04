import inspect

from vector_engine.eval import batch_metrics_summary, retrieval_report_detailed
from vector_engine.ml import kmeans
from vector_engine.training import mine_hard_negatives


def test_kmeans_signature_stability():
    sig = inspect.signature(kmeans)
    assert "vectors" in sig.parameters
    assert "n_clusters" in sig.parameters
    assert "random_state" in sig.parameters
    assert "max_iter" in sig.parameters


def test_mine_hard_negatives_signature_stability():
    sig = inspect.signature(mine_hard_negatives)
    expected = {
        "index",
        "anchors",
        "positives",
        "k",
        "strategy",
        "topk_sample_size",
        "distance_band",
        "exclude_ids",
        "exclude_mask",
        "random_state",
    }
    assert expected.issubset(set(sig.parameters.keys()))


def test_retrieval_eval_signature_stability():
    detailed_sig = inspect.signature(retrieval_report_detailed)
    assert "retrieved_ids" in detailed_sig.parameters
    assert "ground_truth_ids" in detailed_sig.parameters
    assert "ks" in detailed_sig.parameters
    assert "include_per_query" in detailed_sig.parameters

    batch_sig = inspect.signature(batch_metrics_summary)
    assert "reports" in batch_sig.parameters
    assert "include_std" in batch_sig.parameters
