import json
import sys
from pathlib import Path

from .rag_pipeline import run_pipeline


def load_evaluation_set(path="evaluation_set.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate():
    evaluation_set = load_evaluation_set()

    total = len(evaluation_set)
    retrieval_hits = 0
    answer_quality_hits = 0

    print(f"Running evaluation over {total} questions...")
    print("=" * 70)

    for item in evaluation_set:
        question = item["question"]

        result = run_pipeline(question, top_k=3)

        answer = result["answer"]
        citations = result["citations"]

        expected_source = item["expected_source"]
        expected_section = item["expected_section"]

        # Out-of-documents case: expect the refusal phrase, no retrieval.
        if expected_source is None:
            refusal_expected = item.get("expected_answer_contains", "").lower()
            retrieval_hit = not citations
            answer_hit = refusal_expected in answer.lower()
        else:
            # Retrieval quality: is the expected source+section among the citations?
            # Section names in the DB include the medicine name (e.g. "Before you
            # take Doxicap"), so match on the expected words rather than the full string.
            retrieval_hit = any(
                c["source"] == expected_source
                and all(
                    word in c["section"].lower()
                    for word in expected_section.lower().split()
                )
                for c in citations
            )

            # Answer quality: does the answer mention the expected content?
            expected_answer_contains = item.get("expected_answer_contains", "").lower()
            answer_hit = expected_answer_contains in answer.lower()

        if retrieval_hit:
            retrieval_hits += 1
        if answer_hit:
            answer_quality_hits += 1

        print(f"Q: {question}")
        print(f"   Expected source: {expected_source} | section: {expected_section}")
        print(f"   Retrieval hit:   {'YES' if retrieval_hit else 'NO'}")
        print(f"   Answer quality:  {'YES' if answer_hit else 'NO'}")
        print(f"   Answer: {answer[:120]}")
        print("-" * 70)

    print()
    print("=" * 70)
    print(f"Retrieval quality (expected source+section found): "
          f"{retrieval_hits}/{total} = {retrieval_hits / total:.0%}")
    print(f"Answer quality (expected content present):         "
          f"{answer_quality_hits}/{total} = {answer_quality_hits / total:.0%}")


if __name__ == "__main__":
    evaluate()
