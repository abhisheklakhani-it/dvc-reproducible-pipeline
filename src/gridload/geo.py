"""German federal states: population weights and one weather point per state.

Population in millions (Destatis, 31 Dec 2023, rounded). It weights the
weather points and the regional holidays, so a holiday in Bavaria counts
more than one in Bremen.
"""

from __future__ import annotations

# code: (population in millions, latitude, longitude of a large city in the state)
STATES: dict[str, tuple[float, float, float]] = {
    "BW": (11.28, 48.78, 9.18),  # Stuttgart
    "BY": (13.37, 48.14, 11.58),  # Munich
    "BE": (3.76, 52.52, 13.41),  # Berlin
    "BB": (2.57, 52.40, 13.07),  # Potsdam
    "HB": (0.68, 53.08, 8.80),  # Bremen
    "HH": (1.89, 53.55, 9.99),  # Hamburg
    "HE": (6.39, 50.11, 8.68),  # Frankfurt
    "MV": (1.63, 54.09, 12.10),  # Rostock
    "NI": (8.14, 52.37, 9.73),  # Hanover
    "NW": (18.14, 51.46, 7.01),  # Essen (Ruhr area)
    "RP": (4.16, 50.00, 8.27),  # Mainz
    "SL": (0.99, 49.24, 6.99),  # Saarbrücken
    "SN": (4.09, 51.05, 13.74),  # Dresden
    "ST": (2.19, 52.13, 11.63),  # Magdeburg
    "SH": (2.95, 54.32, 10.13),  # Kiel
    "TH": (2.13, 50.98, 11.03),  # Erfurt
}


def weights() -> dict[str, float]:
    """Population share per state; sums to 1."""
    total = sum(pop for pop, _, _ in STATES.values())
    return {code: pop / total for code, (pop, _, _) in STATES.items()}
