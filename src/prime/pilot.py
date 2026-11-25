import math
from statistics import NormalDist

z = NormalDist().inv_cdf

def n_per_group(sigma: float, mde: float, alpha: float = 0.05, power: float = 0.8,
                comparisons: int = 1) -> int:
    """Клиентов на КАЖДУЮ группу, чтобы заметить эффект mde с заданной мощностью."""
    za = z(1 - alpha / (2 * comparisons))
    # Второй квантиль отвечает за мощность. Какой у него аргумент?
    zb = z(power)
    return math.ceil(2 * (za + zb) ** 2 * sigma ** 2 / mde ** 2)


def mde_for_n(sigma: float, n: int, alpha: float = 0.05,
              power: float = 0.8, comparisons: int = 1) -> float:

    za = z(1 - alpha / (2 * comparisons))
    zb = z(power)

    return (za + zb) * sigma * math.sqrt(2 / n)