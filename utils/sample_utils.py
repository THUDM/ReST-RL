import random


def exponential_probability_distribution(n: int, alpha: float = 0.9):
    assert 0 < alpha < 1, "Alpha must be between 0 and 1\n"
    return [alpha ** i for i in range(n)]


def sample_with_exponential_distribution(elements: list, alpha: float = 0.9, k: int = 1) -> list:
    weights = exponential_probability_distribution(len(elements), alpha)
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    return random.choices(elements, weights=normalized_weights, k=k)
