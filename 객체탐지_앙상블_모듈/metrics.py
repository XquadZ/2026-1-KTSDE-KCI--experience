from __future__ import annotations

from typing import Iterable, Sequence, Tuple


def _safe_div(n: float, d: float) -> float:
    return float(n) / float(d) if d else 0.0


def pair_counts(miss_a: Sequence[int], miss_b: Sequence[int]) -> Tuple[int, int, int, int]:
    """
    두 모델의 미검출 이진 벡터(1: miss, 0: detected)에 대해
    N11, N10, N01, N00 카운트를 반환.
    """
    n11 = n10 = n01 = n00 = 0
    for a, b in zip(miss_a, miss_b):
        if a == 1 and b == 1:
            n11 += 1
        elif a == 1 and b == 0:
            n10 += 1
        elif a == 0 and b == 1:
            n01 += 1
        else:
            n00 += 1
    return n11, n10, n01, n00


def disagreement(miss_a: Sequence[int], miss_b: Sequence[int]) -> float:
    """논문 식(3) 불일치도 Dis."""
    n = min(len(miss_a), len(miss_b))
    if n == 0:
        return 0.0
    return sum(1 for a, b in zip(miss_a, miss_b) if a != b) / n


def joint_miss(miss_a: Sequence[int], miss_b: Sequence[int]) -> float:
    """논문 식(4) 동시 미탐지 JMiss."""
    n = min(len(miss_a), len(miss_b))
    if n == 0:
        return 0.0
    return sum(1 for a, b in zip(miss_a, miss_b) if a == 1 and b == 1) / n


def gain_miss(miss_a: Sequence[int], miss_b: Sequence[int]) -> float:
    """논문 식(5) 미검출 개선 기대량 Gain_miss."""
    n = min(len(miss_a), len(miss_b))
    if n == 0:
        return 0.0
    miss_rate_a = sum(miss_a[:n]) / n
    miss_rate_b = sum(miss_b[:n]) / n
    jm = joint_miss(miss_a[:n], miss_b[:n])
    return min(miss_rate_a, miss_rate_b) - jm


def ufp(fp_a: Iterable[str], fp_b: Iterable[str]) -> float:
    """
    논문 식(6) UFP 근사 계산.
    - 입력: 모델별 FP 식별자 집합(예: image_id:x1,y1,x2,y2 형태 문자열)
    """
    set_a = set(fp_a)
    set_b = set(fp_b)
    union = set_a | set_b
    if not union:
        return 0.0
    exclusive = (set_a - set_b) | (set_b - set_a)
    return len(exclusive) / len(union)


def tfc(topk_fp_rate_a: float, topk_fp_rate_b: float) -> float:
    """논문 식(7) TFC: 상위 예측 오답률 평균."""
    return 0.5 * (float(topk_fp_rate_a) + float(topk_fp_rate_b))


def comp(iou_values_on_joint_tp: Sequence[float]) -> float:
    """논문 식(8) Comp: 동시 정탐 샘플의 박스 IoU 평균."""
    if not iou_values_on_joint_tp:
        return 0.0
    return sum(float(v) for v in iou_values_on_joint_tp) / len(iou_values_on_joint_tp)
