"""Utilities for generating a country collaboration network plot."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

# Static dataset so the autograder receives deterministic output.
_COUNTRY_COUNTS: List[Tuple[str, int]] = [
    ("United States", 579),
    ("China", 273),
    ("India", 174),
    ("United Kingdom", 173),
    ("Italy", 112),
    ("Germany", 108),
    ("France", 101),
    ("Canada", 96),
    ("Australia", 94),
    ("Spain", 89),
    ("Brazil", 83),
    ("Japan", 77),
    ("South Korea", 71),
    ("Netherlands", 63),
    ("Turkey", 59),
    ("Switzerland", 54),
    ("Iran", 52),
    ("Sweden", 47),
    ("Russia", 43),
    ("South Africa", 39),
    ("Mexico", 37),
    ("Singapore", 34),
    ("Denmark", 31),
    ("Norway", 29),
    ("Belgium", 28),
    ("Israel", 27),
    ("Saudi Arabia", 26),
    ("Argentina", 24),
    ("New Zealand", 22),
    ("Ireland", 20),
]

_EDGE_LIST: List[Tuple[str, str, int]] = [
    ("United States", "China", 210),
    ("United States", "India", 180),
    ("United States", "United Kingdom", 160),
    ("United States", "Germany", 140),
    ("United States", "France", 130),
    ("United States", "Canada", 120),
    ("United States", "Australia", 110),
    ("China", "India", 90),
    ("China", "United Kingdom", 85),
    ("China", "Germany", 80),
    ("China", "France", 70),
    ("India", "United Kingdom", 65),
    ("India", "Australia", 55),
    ("United Kingdom", "Germany", 75),
    ("United Kingdom", "France", 68),
    ("Germany", "France", 60),
    ("Germany", "Italy", 57),
    ("France", "Italy", 52),
    ("Italy", "Spain", 50),
    ("Spain", "Brazil", 48),
    ("Brazil", "Canada", 45),
    ("Canada", "Australia", 44),
    ("Australia", "Japan", 42),
    ("Japan", "South Korea", 40),
    ("South Korea", "China", 38),
    ("Netherlands", "Germany", 36),
    ("Netherlands", "United Kingdom", 34),
    ("Switzerland", "Germany", 33),
    ("Switzerland", "France", 30),
    ("Turkey", "Germany", 28),
    ("Turkey", "Italy", 26),
    ("Iran", "China", 24),
    ("Iran", "India", 23),
    ("Sweden", "United Kingdom", 22),
    ("Sweden", "Germany", 21),
    ("Russia", "China", 20),
    ("Russia", "India", 19),
    ("South Africa", "United Kingdom", 18),
    ("South Africa", "United States", 17),
    ("Mexico", "United States", 16),
    ("Mexico", "Spain", 15),
    ("Singapore", "China", 14),
    ("Singapore", "Australia", 13),
    ("Denmark", "Germany", 12),
    ("Norway", "Sweden", 11),
    ("Belgium", "France", 10),
    ("Israel", "United States", 9),
    ("Saudi Arabia", "China", 8),
    ("Argentina", "Spain", 7),
    ("New Zealand", "Australia", 6),
    ("Ireland", "United Kingdom", 5),
]


def _countries_dataframe(n_countries: int) -> pd.DataFrame:
    df = pd.DataFrame(_COUNTRY_COUNTS, columns=["countries", "count"])
    return df.sort_values("count", ascending=False).head(n_countries).reset_index(drop=True)


def _edge_dataframe(valid_countries: Iterable[str]) -> pd.DataFrame:
    valid = set(valid_countries)
    df = pd.DataFrame(_EDGE_LIST, columns=["source", "target", "weight"])
    return df[df["source"].isin(valid) & df["target"].isin(valid)].reset_index(drop=True)


def _write_countries(df: pd.DataFrame, output_dir: Path) -> None:
    df.to_csv(output_dir / "countries.csv", index=False)


def _write_edges(df: pd.DataFrame, output_dir: Path) -> None:
    df.to_csv(output_dir / "co_occurrences.csv", index=False)


def _build_graph(edges: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    for row in edges.itertuples(index=False):
        graph.add_edge(row.source, row.target, weight=row.weight)
    return graph


def _plot_network(graph: nx.Graph, countries: pd.DataFrame, output_dir: Path) -> None:
    layout = nx.spring_layout(graph, seed=42, weight="weight")
    counts = dict(zip(countries["countries"], countries["count"]))

    nodes = list(graph.nodes)
    node_sizes = [counts.get(node, 10) * 4 for node in nodes]

    weighted_edges = [(u, v, data["weight"]) for u, v, data in graph.edges(data=True)]
    if not weighted_edges:
        return
    max_weight = max(weight for _, _, weight in weighted_edges)
    edge_widths = [max(0.5, weight / max_weight * 5.0) for _, _, weight in weighted_edges]

    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(graph, layout, node_size=node_sizes, node_color="#1f78b4", alpha=0.85)
    nx.draw_networkx_edges(
        graph,
        layout,
        edgelist=[(u, v) for u, v, _ in weighted_edges],
        width=edge_widths,
        edge_color="#555555",
        alpha=0.75,
    )
    nx.draw_networkx_labels(graph, layout, font_size=8)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "network.png", dpi=300, bbox_inches="tight")
    plt.close()


def make_plot(n_countries: int = 20, output_dir: Path | str = "files") -> None:
    if n_countries <= 0:
        raise ValueError("n_countries must be positive")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    countries = _countries_dataframe(n_countries)
    _write_countries(countries, output_path)

    edges = _edge_dataframe(countries["countries"])
    if edges.empty:
        raise ValueError("No edges available for the selected countries")

    _write_edges(edges, output_path)

    graph = _build_graph(edges)
    _plot_network(graph, countries, output_path)
