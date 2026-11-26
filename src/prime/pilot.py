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


import pandas as pd


def diff_ci(t: pd.Series, c: pd.Series, alpha: float = 0.05) -> dict:
    """Разность средних test − control, доверительный интервал и p-value."""

    # Разность средних
    d = t.mean() - c.mean()

    # Стандартная ошибка разности средних
    se = (t.var() / len(t) + c.var() / len(c)) ** 0.5

    # Критическое значение
    q = NormalDist().inv_cdf(1 - alpha / 2)

    # Двустороннее p-value
    p = 2 * (1 - NormalDist().cdf(abs(d / se)))

    return {
        "diff": d,
        "lo": d - q * se,
        "hi": d + q * se,
        "p": p
    }


def srm_pvalue(sizes) -> float:
    """p-value критерия хи-квадрат для двух или трёх равных групп."""

    # Общее количество клиентов
    n = sum(sizes)

    # Ожидаемое количество клиентов в каждой группе
    exp = n / len(sizes)

    # Статистика хи-квадрат
    chi2 = sum((s - exp) ** 2 / exp for s in sizes)

    # Две группы
    if len(sizes) == 2:
        return math.erfc(math.sqrt(chi2 / 2))

    # Три группы
    if len(sizes) == 3:
        return math.exp(-chi2 / 2)

    raise ValueError("для 4 и более групп — scipy.stats.chisquare")


def cuped(y, x):
    """Возвращает скорректированную метрику и коэффициент theta."""

    # Коэффициент корректировки
    theta = y.cov(x) / x.var()

    # Скорректированная метрика и коэффициент theta
    return y - theta * (x - x.mean()), theta