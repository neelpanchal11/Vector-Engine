import numpy as np

from vector_engine.eval import batch_metrics_summary, retrieval_cohort_report, retrieval_report_detailed


def test_retrieval_report_detailed_error_buckets_present():
    retrieved = np.array([["a", "b", "c"], ["m", "n", "o"]], dtype=object)
    gt = [["z"], ["n"]]
    out = retrieval_report_detailed(retrieved, gt, ks=(1, 2), include_per_query=False, include_error_buckets=True)
    assert "summary" in out
    assert "error_buckets" in out
    assert "zero_hit_rate@1" in out["error_buckets"]


def test_batch_metrics_summary_with_ci():
    reports = [
        {"recall@1": 0.5, "ndcg@1": 0.6},
        {"recall@1": 0.7, "ndcg@1": 0.8},
        {"recall@1": 0.9, "ndcg@1": 1.0},
    ]
    out = batch_metrics_summary(reports, include_std=True, include_ci=True, n_bootstrap=100, random_state=7)
    assert "recall@1_std" in out
    assert "recall@1_ci_lower" in out
    assert "recall@1_ci_upper" in out
    assert out["recall@1_ci_lower"] <= out["recall@1"] <= out["recall@1_ci_upper"]


def test_retrieval_cohort_report():
    retrieved = np.array(
        [
            ["a", "b", "c"],
            ["m", "n", "o"],
            ["x", "y", "z"],
            ["k", "l", "m"],
        ],
        dtype=object,
    )
    gt = [["a"], ["n"], ["q"], ["k"]]
    cohorts = ["easy", "easy", "hard", "hard"]
    out = retrieval_cohort_report(retrieved, gt, cohorts, ks=(1, 2))
    assert "overall" in out
    assert "per_cohort" in out
    assert "easy" in out["per_cohort"]
    assert "hard" in out["per_cohort"]


def test_randomized_metric_bounds_property():
    rng = np.random.default_rng(11)
    for _ in range(10):
        n = 6
        k = 4
        universe = [f"id-{i}" for i in range(20)]
        retrieved = np.asarray(
            [[universe[int(x)] for x in rng.choice(len(universe), size=k, replace=False)] for _ in range(n)],
            dtype=object,
        )
        gt = [[universe[int(x)] for x in rng.choice(len(universe), size=2, replace=False)] for _ in range(n)]
        detail = retrieval_report_detailed(retrieved, gt, ks=(1, k), include_per_query=False)
        for _, value in detail["summary"].items():
            assert 0.0 <= value <= 1.0


def test_retrieval_error_buckets_ignore_empty_ground_truth_for_hit_rates():
    retrieved = np.array(
        [
            ["a", "x", "y"],  # perfect hit at k=1
            ["z", "y", "x"],  # zero hit at k=1
            ["m", "n", "o"],  # ignored for hit-rates (empty ground truth)
        ],
        dtype=object,
    )
    gt = [["a"], ["b"], []]
    out = retrieval_report_detailed(retrieved, gt, ks=(1,), include_per_query=False, include_error_buckets=True)
    buckets = out["error_buckets"]
    assert buckets["zero_hit_rate@1"] == 0.5
    assert buckets["perfect_recall_rate@1"] == 0.5
    assert buckets["no_ground_truth_rate@1"] == (1.0 / 3.0)
