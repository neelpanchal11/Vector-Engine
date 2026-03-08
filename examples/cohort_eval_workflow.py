from __future__ import annotations

import numpy as np

from vector_engine.eval import retrieval_cohort_report


def main() -> None:
    retrieved = np.array(
        [
            ["doc-a", "doc-b", "doc-c"],
            ["doc-d", "doc-e", "doc-f"],
            ["doc-x", "doc-y", "doc-z"],
            ["doc-k", "doc-l", "doc-m"],
        ],
        dtype=object,
    )
    ground_truth = [
        ["doc-a"],
        ["doc-e"],
        ["doc-missing"],
        ["doc-k"],
    ]
    cohorts = ["faq", "faq", "long_tail", "long_tail"]
    report = retrieval_cohort_report(retrieved, ground_truth, cohorts, ks=(1, 3))
    print("overall", report["overall"])
    print("cohort_sizes", report["cohort_sizes"])
    print("per_cohort", report["per_cohort"])


if __name__ == "__main__":
    main()
