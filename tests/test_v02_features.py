import numpy as np
import pytest

from vector_engine import VectorArray, VectorIndex
from vector_engine.eval import batch_metrics_summary, retrieval_cohort_report, retrieval_report_detailed
from vector_engine.ml import kmeans
from vector_engine.training import mine_hard_negatives


def test_kmeans_returns_rich_result():
    pytest.importorskip("sklearn")
    x = np.array(
        [
            [0.0, 0.0],
            [0.1, -0.1],
            [3.0, 3.0],
            [3.2, 2.9],
        ],
        dtype=np.float32,
    )
    va = VectorArray.from_numpy(x, ids=np.arange(len(x)))
    result_a = kmeans(va, n_clusters=2, random_state=7)
    result_b = kmeans(va, n_clusters=2, random_state=7)
    assert result_a.labels.shape == (4,)
    assert result_a.centers.shape == (2, 2)
    assert result_a.inertia >= 0.0
    assert np.array_equal(result_a.labels, result_b.labels)


def test_kmeans_validates_seed_and_values():
    pytest.importorskip("sklearn")
    x = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    va = VectorArray.from_numpy(x, ids=np.arange(len(x)))
    with pytest.raises(TypeError, match="ml_error"):
        kmeans(va, n_clusters=2, random_state=7.5)  # type: ignore[arg-type]

    x_bad = x.copy()
    x_bad[0, 0] = np.nan
    va_bad = VectorArray.from_numpy(x_bad, ids=np.arange(len(x_bad)))
    with pytest.raises(ValueError, match="ml_error"):
        kmeans(va_bad, n_clusters=2)


def test_hard_negative_strategies_and_exclusions():
    xb = VectorArray.from_numpy(np.random.randn(30, 8).astype(np.float32), ids=np.arange(30), normalize=True)
    index = VectorIndex.create(xb, metric="cosine", backend="bruteforce")
    anchors = xb.subset([0, 1, 2, 3])
    positives = np.asarray([0, 1, 2, 3], dtype=object)
    excluded = {4, 5}

    top1 = mine_hard_negatives(index, anchors, positives=positives, k=8, strategy="top1", exclude_ids=excluded)
    sampled = mine_hard_negatives(
        index,
        anchors,
        positives=positives,
        k=8,
        strategy="topk_sample",
        topk_sample_size=3,
        exclude_ids=excluded,
        random_state=123,
    )
    band = mine_hard_negatives(
        index,
        anchors,
        positives=positives,
        k=8,
        strategy="distance_band",
        distance_band=(1, 4),
        random_state=5,
    )

    assert len(top1.negatives) == 4
    assert len(sampled.negatives) == 4
    assert len(band.negatives) == 4
    for arr in [top1.negatives, sampled.negatives, band.negatives]:
        for p, n in zip(positives, arr):
            assert p != n


def test_hard_negative_validates_strategy_parameters():
    xb = VectorArray.from_numpy(np.random.randn(20, 8).astype(np.float32), ids=np.arange(20), normalize=True)
    index = VectorIndex.create(xb, metric="cosine", backend="bruteforce")
    anchors = xb.subset([0, 1])
    positives = np.asarray([0, 1], dtype=object)

    with pytest.raises(ValueError, match="training_error"):
        mine_hard_negatives(index, anchors, positives=positives, strategy="topk_sample", topk_sample_size=0)
    with pytest.raises(ValueError, match="training_error"):
        mine_hard_negatives(index, anchors, positives=positives, strategy="distance_band", distance_band=(2, 2))


def test_hard_negative_topk_sample_is_seeded_and_stable():
    xb = VectorArray.from_numpy(np.random.randn(40, 16).astype(np.float32), ids=np.arange(40), normalize=True)
    index = VectorIndex.create(xb, metric="cosine", backend="bruteforce")
    anchors = xb.subset([0, 1, 2, 3, 4, 5])
    positives = np.asarray([0, 1, 2, 3, 4, 5], dtype=object)
    run_a = mine_hard_negatives(
        index,
        anchors,
        positives=positives,
        k=10,
        strategy="topk_sample",
        topk_sample_size=4,
        random_state=19,
    )
    run_b = mine_hard_negatives(
        index,
        anchors,
        positives=positives,
        k=10,
        strategy="topk_sample",
        topk_sample_size=4,
        random_state=19,
    )
    assert np.array_equal(run_a.negatives, run_b.negatives)
    assert np.allclose(run_a.negative_scores, run_b.negative_scores)


def test_eval_detailed_and_batch_summary():
    retrieved = np.array([["a", "b", "c"], ["m", "n", "o"]], dtype=object)
    gt = [["a", "z"], ["x", "n"]]
    detail = retrieval_report_detailed(retrieved, gt, ks=(1, 2))
    assert "summary" in detail
    assert "per_query" in detail
    assert len(detail["per_query"]) == 2

    compact = retrieval_report_detailed(retrieved, gt, ks=(1, 2), include_per_query=False)
    assert "per_query" not in compact

    summary = batch_metrics_summary([detail["summary"], detail["summary"]], include_std=True)
    assert "recall@2" in summary
    assert "recall@2_std" in summary
    assert summary["num_reports"] == 2.0


def test_eval_validation_on_malformed_inputs():
    retrieved = np.array([["a", "b"]], dtype=object)
    gt_bad = [["a"], ["b"]]
    with pytest.raises(ValueError, match="eval_error"):
        retrieval_report_detailed(retrieved, gt_bad, ks=(1,))
    with pytest.raises(TypeError, match="eval_error"):
        retrieval_report_detailed(retrieved, ["abc"], ks=(1,))  # type: ignore[list-item]
    with pytest.raises(ValueError, match="eval_error"):
        retrieval_report_detailed(retrieved, [["a"]], ks=(3,))


def test_eval_cohort_report_shapes_and_counts():
    retrieved = np.array([["a", "b"], ["x", "y"], ["m", "n"]], dtype=object)
    gt = [["a"], ["z"], ["m"]]
    cohorts = ["head", "tail", "tail"]
    report = retrieval_cohort_report(retrieved, gt, cohorts, ks=(1, 2))
    assert set(report.keys()) == {"overall", "cohort_sizes", "per_cohort"}
    assert report["cohort_sizes"]["head"] == 1
    assert report["cohort_sizes"]["tail"] == 2
    assert "recall@1" in report["overall"]


def test_eval_cohort_report_validates_input_lengths():
    retrieved = np.array([["a", "b"], ["x", "y"]], dtype=object)
    gt = [["a"], ["x"]]
    with pytest.raises(ValueError, match="eval_error"):
        retrieval_cohort_report(retrieved, gt, ["only_one"], ks=(1, 2))
