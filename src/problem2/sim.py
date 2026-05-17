from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

matplotlib.use("Agg")

plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC", "Noto Sans CJK JP",
    "Noto Sans CJK TC", "WenQuanYi Micro Hei",
    "SimHei", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

NUM_GROUPS = 16
GROUP_SIZE = 4

_CITY_SUFFIX = "市"
_TEAM_SUFFIX = "队"


def load_teams(csv_path: str | Path) -> tuple[list[str], dict[str, list[str]]]:
    with open(csv_path, encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    cities: list[str] = []
    counties: dict[str, list[str]] = defaultdict(list)

    for row in rows:
        name = row["team_name"].removesuffix(_TEAM_SUFFIX)
        parent = row["parent_city"] + _CITY_SUFFIX
        if row["level"] == "city":
            cities.append(name)
        else:
            counties[parent].append(name)

    return cities, dict(counties)


def _all_counties(cities: list[str], county_teams: dict[str, list[str]]) -> list[tuple[str, str]]:
    return [(county, city) for city in cities for county in county_teams[city]]


def _validate(
    groups: dict[int, list[str]],
    cities: list[str],
    city_group: dict[str, int],
    county_teams: dict[str, list[str]],
) -> None:
    for members in groups.values():
        assert len(members) == GROUP_SIZE

    city_gids = [city_group[c] for c in cities]
    assert len(set(city_gids)) == len(cities)

    for city, gid in city_group.items():
        for county in county_teams[city]:
            assert county not in groups[gid]


def draw_once(
    cities: list[str],
    county_teams: dict[str, list[str]],
    seed: int | None = None,
) -> tuple[dict[int, list[str]], dict[str, int]]:
    if seed is not None:
        random.seed(seed)

    num_city = len(cities)
    all_counties = _all_counties(cities, county_teams)
    groups: dict[int, list[str]] = {i: [] for i in range(1, NUM_GROUPS + 1)}
    city_group_ids = random.sample(list(groups.keys()), num_city)
    shuffled_cities = random.sample(cities, num_city)
    city_group = {city: gid for city, gid in zip(shuffled_cities, city_group_ids)}

    for city, gid in city_group.items():
        groups[gid].append(city)

    all_counties_shuffled = list(all_counties)
    random.shuffle(all_counties_shuffled)

    for _ in range(100):
        for gid in groups:
            groups[gid] = [t for t in groups[gid] if t in cities]

        failed = False
        for county, city in all_counties_shuffled:
            city_gid = city_group[city]
            strict: list[int] = []
            relaxed: list[int] = []

            for gid in range(1, NUM_GROUPS + 1):
                if gid == city_gid or len(groups[gid]) >= GROUP_SIZE:
                    continue
                has_same = any(t in county_teams[city] for t in groups[gid])
                if not has_same:
                    strict.append(gid)
                relaxed.append(gid)

            if strict:
                chosen = random.choice(strict)
            elif relaxed:
                chosen = random.choice(relaxed)
            else:
                failed = True
                break

            groups[chosen].append(county)

        if not failed:
            break

    _validate(groups, cities, city_group, county_teams)
    return groups, city_group


def run_simulations(
    cities: list[str],
    county_teams: dict[str, list[str]],
    n: int = 10000,
) -> dict[str, list[int]]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0] * NUM_GROUPS)

    for sim_idx in range(n):
        groups, city_group = draw_once(cities, county_teams, seed=sim_idx)
        for gid, members in groups.items():
            for team in members:
                counts[team][gid - 1] += 1

    return dict(counts)


def chi_square_test(
    observed: list[int], expected_probs: list[float], n_total: int,
) -> tuple[float, float]:
    valid = [i for i, p in enumerate(expected_probs) if p > 0]
    if len(valid) <= 1:
        return 0.0, 1.0
    obs = np.array([observed[i] for i in valid])
    exp = np.array([expected_probs[i] * n_total for i in valid])
    chi2, p_value = stats.chisquare(f_obs=obs, f_exp=exp)
    return float(chi2), float(p_value)


def collect_p_values(
    team_group_counts: dict[str, list[int]],
    cities: list[str],
    county_teams: dict[str, list[str]],
    n_sim: int,
) -> tuple[list[float], list[float]]:
    expected = [1.0 / NUM_GROUPS] * NUM_GROUPS

    city_pv = [chi_square_test(team_group_counts[c], expected, n_sim)[1] for c in cities]
    county_pv: list[float] = []
    for city in cities:
        for county in county_teams[city]:
            county_pv.append(chi_square_test(team_group_counts[county], expected, n_sim)[1])

    return city_pv, county_pv


def plot_results(
    team_group_counts: dict[str, list[int]],
    cities: list[str],
    county_teams: dict[str, list[str]],
    n_sim: int,
    all_p_values: list[float],
    output_dir: str | Path,
) -> None:
    num_city = len(cities)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    ax1 = axes[0, 0]
    city_probs = [[c / n_sim for c in team_group_counts[city]] for city in cities]
    im1 = ax1.imshow(city_probs, cmap="YlOrRd", aspect="auto", vmin=0, vmax=2.0 / NUM_GROUPS)
    ax1.set_xticks(range(NUM_GROUPS))
    ax1.set_xticklabels([f"G{i+1}" for i in range(NUM_GROUPS)], rotation=45)
    ax1.set_yticks(range(num_city))
    ax1.set_yticklabels(cities)
    ax1.set_title(f"市级队落入各组的概率 (N={n_sim})\n期望 = 1/16 ≈ {1/16:.4f}")
    plt.colorbar(im1, ax=ax1, label="概率")

    ax2 = axes[0, 1]
    ax2.hist(all_p_values, bins=30, edgecolor="black", alpha=0.7, density=True)
    ax2.axhline(1.0, color="red", linestyle="--", label="均匀分布期望")
    ax2.axvline(0.05, color="orange", linestyle=":", label="α=0.05")
    ax2.set_xlabel("p 值")
    ax2.set_ylabel("密度")
    ax2.set_title(f"卡方检验 p 值分布\n平均 p = {np.mean(all_p_values):.3f}")
    ax2.legend()

    ax3 = axes[1, 0]
    sample_counties = [county_teams[city][0] for city in cities]
    county_probs = [[c / n_sim for c in team_group_counts[co]] for co in sample_counties]
    im3 = ax3.imshow(county_probs, cmap="YlOrRd", aspect="auto", vmin=0, vmax=2.0 / NUM_GROUPS)
    ax3.set_xticks(range(NUM_GROUPS))
    ax3.set_xticklabels([f"G{i+1}" for i in range(NUM_GROUPS)], rotation=45)
    ax3.set_yticks(range(len(sample_counties)))
    ax3.set_yticklabels(sample_counties)
    ax3.set_title(f"代表性县级队落入各组的概率 (N={n_sim})\n期望 = 1/16 ≈ {1/16:.4f}")
    plt.colorbar(im3, ax=ax3, label="概率")

    ax4 = axes[1, 1]
    stats.probplot(all_p_values, dist="uniform", plot=ax4)
    ax4.set_title("p值 Q-Q 图 (vs 均匀分布)")
    ax4.get_lines()[0].set_markerfacecolor("blue")
    ax4.get_lines()[0].set_markeredgecolor("blue")
    ax4.get_lines()[0].set_markersize(3)
    ax4.get_lines()[1].set_color("red")

    plt.tight_layout()
    plt.savefig(output_dir / "fairness_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig2, ax = plt.subplots(figsize=(14, 8))
    groups, _ = draw_once(cities, county_teams, seed=42)
    table_data = [[f"G{gid}"] + groups[gid] for gid in range(1, NUM_GROUPS + 1)]
    ax.axis("off")
    table = ax.table(
        cellText=table_data,
        colLabels=["小组"] + [f"队{i}" for i in range(1, 5)],
        cellLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax.set_title("单次抽签示例 (seed=42)", fontsize=14, fontweight="bold", pad=20)
    plt.savefig(output_dir / "draw_example.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)


def print_summary(
    city_p_values: list[float], county_p_values: list[float], n_sim: int,
) -> None:
    all_pv = city_p_values + county_p_values

    print(f"检验队伍总数: {len(all_pv)}")
    print(f"平均 p 值: {np.mean(all_pv):.4f}")
    print(f"p < 0.01: {sum(1 for p in all_pv if p < 0.01)}")
    print(f"p < 0.05: {sum(1 for p in all_pv if p < 0.05)}")
    print(f"期望 p<0.05 数 (纯随机): {0.05 * len(all_pv):.1f}")


def main(csv_path: str | Path, output_dir: str | Path, n_sim: int = 10000) -> None:
    cities, county_teams = load_teams(csv_path)

    groups, _ = draw_once(cities, county_teams, seed=42)
    for gid in range(1, NUM_GROUPS + 1):
        labeled = [f"【{m}】" if m in cities else m for m in groups[gid]]
        print(f"G{gid:02d}: {', '.join(labeled)}")

    team_group_counts = run_simulations(cities, county_teams, n_sim)
    city_pv, county_pv = collect_p_values(team_group_counts, cities, county_teams, n_sim)
    plot_results(team_group_counts, cities, county_teams, n_sim, city_pv + county_pv, output_dir)
    print_summary(city_pv, county_pv, n_sim)
