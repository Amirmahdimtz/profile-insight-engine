from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar

from src.core.profile_analysis.contracts import ContractValidationError, validate_observable_claim

from evaluation.contracts import EvaluationValidationError


T = TypeVar("T", bound=Hashable)


@dataclass(frozen=True)
class PrecisionRecallF1:
    precision: float
    recall: float
    f1: float


def _levenshtein_distance(reference: Sequence[T], prediction: Sequence[T]) -> int:
    if len(reference) < len(prediction):
        reference, prediction = prediction, reference
    previous = list(range(len(prediction) + 1))
    for reference_index, reference_item in enumerate(reference, start=1):
        current = [reference_index]
        for prediction_index, prediction_item in enumerate(prediction, start=1):
            insertion = current[prediction_index - 1] + 1
            deletion = previous[prediction_index] + 1
            substitution = previous[prediction_index - 1] + (
                0 if reference_item == prediction_item else 1
            )
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def character_error_rate(reference: str, prediction: str) -> float:
    """Unicode code-point CER; no case-folding or Unicode normalization is applied."""

    if not isinstance(reference, str) or not isinstance(prediction, str):
        raise EvaluationValidationError("CER reference and prediction must be strings")
    if not reference:
        return 0.0 if not prediction else 1.0
    return _levenshtein_distance(tuple(reference), tuple(prediction)) / len(reference)


def word_error_rate(reference: str, prediction: str) -> float:
    """Whitespace-token WER without lowercasing or Unicode normalization."""

    if not isinstance(reference, str) or not isinstance(prediction, str):
        raise EvaluationValidationError("WER reference and prediction must be strings")
    reference_words = reference.split()
    prediction_words = prediction.split()
    if not reference_words:
        return 0.0 if not prediction_words else 1.0
    return _levenshtein_distance(reference_words, prediction_words) / len(reference_words)


def precision_recall_f1(expected: Iterable[T], predicted: Iterable[T]) -> PrecisionRecallF1:
    expected_set = set(expected)
    predicted_set = set(predicted)
    true_positive = len(expected_set & predicted_set)

    if predicted_set:
        precision = true_positive / len(predicted_set)
    else:
        precision = 1.0 if not expected_set else 0.0

    if expected_set:
        recall = true_positive / len(expected_set)
    else:
        recall = 1.0 if not predicted_set else 0.0

    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return PrecisionRecallF1(precision=precision, recall=recall, f1=f1)


def macro_f1(
    expected_by_sample: Sequence[Iterable[T]],
    predicted_by_sample: Sequence[Iterable[T]],
    *,
    classes: Iterable[T] | None = None,
) -> float:
    if len(expected_by_sample) != len(predicted_by_sample):
        raise EvaluationValidationError(
            "macro F1 expected and predicted sample collections must have equal length"
        )

    expected_sets = [set(items) for items in expected_by_sample]
    predicted_sets = [set(items) for items in predicted_by_sample]

    if classes is None:
        class_set: set[T] = set()
        for items in expected_sets:
            class_set.update(items)
        for items in predicted_sets:
            class_set.update(items)
    else:
        class_set = set(classes)
        unexpected_predictions = set().union(*predicted_sets) - class_set if predicted_sets else set()
        unexpected_expected = set().union(*expected_sets) - class_set if expected_sets else set()
        if unexpected_predictions or unexpected_expected:
            raise EvaluationValidationError(
                "macro F1 inputs contain labels outside the explicit class universe"
            )

    supported_classes = [
        label for label in class_set if any(label in expected for expected in expected_sets)
    ]
    if not supported_classes:
        has_predictions = any(predicted for predicted in predicted_sets)
        return 0.0 if has_predictions else 1.0

    per_class_f1: list[float] = []
    for label in sorted(supported_classes, key=repr):
        expected_binary = [label in items for items in expected_sets]
        predicted_binary = [label in items for items in predicted_sets]
        true_positive = sum(e and p for e, p in zip(expected_binary, predicted_binary))
        false_positive = sum((not e) and p for e, p in zip(expected_binary, predicted_binary))
        false_negative = sum(e and (not p) for e, p in zip(expected_binary, predicted_binary))
        denominator = 2 * true_positive + false_positive + false_negative
        per_class_f1.append(0.0 if denominator == 0 else (2 * true_positive) / denominator)
    return sum(per_class_f1) / len(per_class_f1)


def recall_at_k(relevant: Iterable[T], ranked: Sequence[T], k: int) -> float:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise EvaluationValidationError("retrieval k must be a positive integer")
    ranked_items = tuple(ranked)
    if len(ranked_items) != len(set(ranked_items)):
        raise EvaluationValidationError("retrieval ranking must not contain duplicate items")
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    return len(relevant_set & set(ranked_items[:k])) / len(relevant_set)


def average_precision(relevant: Iterable[T], ranked: Sequence[T]) -> float:
    ranked_items = tuple(ranked)
    if len(ranked_items) != len(set(ranked_items)):
        raise EvaluationValidationError("retrieval ranking must not contain duplicate items")
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for index, item in enumerate(ranked_items, start=1):
        if item not in relevant_set:
            continue
        hits += 1
        precision_sum += hits / index
    return precision_sum / len(relevant_set)


def mean_average_precision(
    relevant_by_query: Mapping[T, Iterable[T]],
    ranked_by_query: Mapping[T, Sequence[T]],
) -> float:
    if set(relevant_by_query) != set(ranked_by_query):
        raise EvaluationValidationError(
            "mAP relevant and ranked mappings must contain the same query ids"
        )
    if not relevant_by_query:
        return 0.0
    values = [
        average_precision(relevant_by_query[query], ranked_by_query[query])
        for query in sorted(relevant_by_query, key=repr)
    ]
    return sum(values) / len(values)


def expected_calibration_error(
    observations: Sequence[tuple[float, bool]], *, bins: int
) -> float:
    if isinstance(bins, bool) or not isinstance(bins, int) or bins <= 0:
        raise EvaluationValidationError("calibration bins must be a positive integer")
    if not observations:
        return 0.0

    bucket_confidences: list[list[float]] = [[] for _ in range(bins)]
    bucket_correctness: list[list[bool]] = [[] for _ in range(bins)]
    for confidence, is_correct in observations:
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise EvaluationValidationError("calibration confidence must be numeric")
        confidence = float(confidence)
        if not math.isfinite(confidence) or confidence < 0.0 or confidence > 1.0:
            raise EvaluationValidationError(
                "calibration confidence must be finite and in [0, 1]"
            )
        if not isinstance(is_correct, bool):
            raise EvaluationValidationError("calibration correctness must be boolean")
        bucket_index = min(int(confidence * bins), bins - 1)
        bucket_confidences[bucket_index].append(confidence)
        bucket_correctness[bucket_index].append(is_correct)

    total = len(observations)
    error = 0.0
    for confidences, correctness in zip(bucket_confidences, bucket_correctness):
        if not confidences:
            continue
        average_confidence = sum(confidences) / len(confidences)
        accuracy = sum(correctness) / len(correctness)
        error += (len(confidences) / total) * abs(accuracy - average_confidence)
    return error


def unsupported_claim_rate(claims: Sequence[tuple[str, str]]) -> float:
    if not claims:
        return 0.0
    unsupported = 0
    for key, label in claims:
        if (
            not isinstance(key, str)
            or not key
            or key != key.strip()
            or not isinstance(label, str)
            or not label
            or label != label.strip()
        ):
            raise EvaluationValidationError(
                "claim key and label must be non-empty trimmed strings"
            )
        try:
            validate_observable_claim(key, label)
        except ContractValidationError:
            unsupported += 1
    return unsupported / len(claims)
