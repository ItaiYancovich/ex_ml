"""Checks for the area-proportional circle chart.

Run with ``python test_circle_area_chart.py`` (or pytest, if it is installed).

The two things that have to be true are that the areas really are proportional
to the numbers, and that no label spills out of the cell it belongs to.
"""

from __future__ import annotations

import math
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np

import circle_area_chart as cac
from voronoi_treemap import (
    circle_boundary,
    pole_of_inaccessibility,
    polygon_area,
    voronoi_treemap,
    worst_roundness,
)

DATASETS = {
    "three": [5.0, 3.0, 2.0],
    "one": [7.0],
    "equal": [1.0] * 12,
    "skewed": [1900, 573, 451, 204, 145, 92, 87, 86, 82, 77, 35, 22, 19, 17],
    "lognormal": list(np.round(np.random.default_rng(4).lognormal(0, 1.2, 30), 3)),
}


def _points_inside(poly, pts):
    """Vectorised point-in-convex-polygon test (counter-clockwise polygon)."""
    a, b = poly, np.roll(poly, -1, axis=0)
    pts = np.atleast_2d(pts)
    cross = ((b[:, 0] - a[:, 0])[None, :] * (pts[:, 1, None] - a[None, :, 1])
             - (b[:, 1] - a[:, 1])[None, :] * (pts[:, 0, None] - a[None, :, 0]))
    return (cross >= -1e-7).all(axis=1)


def test_areas_are_proportional():
    for name, values in DATASETS.items():
        result = voronoi_treemap(values, radius=1.0, seed=1, tol=0.004)
        share = result.areas / result.areas.sum()
        want = np.asarray(values, dtype=float) / np.sum(values)
        worst = np.max(np.abs(share - want) / want)
        assert worst < 0.02, f"{name}: worst area error {worst:.2%}"
        # the cells have to fill the circle exactly, with no double counting
        assert math.isclose(result.areas.sum(), math.pi, rel_tol=2e-3), name


def test_cells_are_disjoint_and_non_empty():
    values = DATASETS["skewed"]
    result = voronoi_treemap(values, radius=1.0, seed=2)
    for verts in result.cells:
        assert len(verts) >= 3
        assert polygon_area(verts) > 0  # counter-clockwise, non-degenerate
    # every site falls inside its own cell, which is what makes them disjoint
    for site, verts in zip(result.sites, result.cells):
        assert _points_inside(verts, site)[0]
    assert worst_roundness(result.cells) > 0.15


def test_cells_stay_inside_the_boundary():
    boundary = circle_boundary(radius=1.0)
    # the boundary is a 512-gon scaled to have exactly the circle's area, so it
    # pokes a hair outside r = 1; the real invariant is against that polygon
    assert math.isclose(polygon_area(boundary), math.pi, rel_tol=1e-9)
    result = voronoi_treemap(DATASETS["equal"], boundary=boundary, seed=0)
    for verts in result.cells:
        assert _points_inside(boundary, verts).all()
        assert np.linalg.norm(verts, axis=1).max() <= 1.001


def test_layout_is_reproducible():
    a = voronoi_treemap(DATASETS["skewed"], seed=5)
    b = voronoi_treemap(DATASETS["skewed"], seed=5)
    for x, y in zip(a.cells, b.cells):
        assert np.allclose(x, y)


def test_pole_of_inaccessibility_is_inside():
    result = voronoi_treemap(DATASETS["skewed"], seed=3)
    for verts in result.cells:
        point, radius = pole_of_inaccessibility(verts)
        assert radius > 0
        assert _points_inside(verts, point)[0]
        # the reported radius must really be free space
        angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        ring = point + radius * 0.99 * np.column_stack([np.cos(angles), np.sin(angles)])
        assert _points_inside(verts, ring).all()


def test_rejects_bad_input():
    for bad in ([], [1, 0, 2], [1, -3], [1, float("nan")]):
        try:
            voronoi_treemap(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")


def test_labels_stay_inside_their_cells():
    values = [1900, 573, 451, 204, 145, 92, 87, 86, 82, 77, 56, 35, 22, 19, 17]
    labels = [f"Country {i}" for i in range(len(values))]
    fig, ax = cac.circle_area_chart(values, labels, title="Fit check")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    cells = [ax.transData.transform(c) for c in ax._treemap_result.cells]

    checked = 0
    for text in ax.texts:
        box = text.get_window_extent(renderer=renderer)
        anchor = ax.transData.transform(text.get_position())
        host = [c for c in cells if _points_inside(c, anchor)[0]]
        assert host, f"label {text.get_text()!r} is not inside any cell"
        corners = np.array([[box.x0, box.y0], [box.x1, box.y0],
                            [box.x0, box.y1], [box.x1, box.y1]])
        assert _points_inside(host[0], corners).all(), \
            f"label {text.get_text()!r} overflows its cell"
        checked += 1
    assert checked >= len(values), "expected a name and a value per cell"


def test_colour_modes_and_themes():
    values = [5, 4, 3, 2, 1]
    groups = ["a", "a", "b", "b", "c"]
    for theme in ("light", "dark"):
        fig, ax = cac.circle_area_chart(values, groups=groups, theme=theme)
        patches = [p for p in ax.patches if hasattr(p, "get_xy")]
        assert len(patches) == len(values)
        fig, ax = cac.circle_area_chart(values, color_by="single", theme=theme)
        faces = {tuple(p.get_facecolor()) for p in ax.patches[:len(values)]}
        assert len(faces) == 1


def test_group_order_pins_colours():
    values = [5, 4, 3, 2, 1]
    groups = ["a", "a", "b", "b", "c"]
    order = ["c", "b", "a"]
    fig, ax = cac.circle_area_chart(values, groups=groups, group_order=order)
    first = ax.patches[4].get_facecolor()  # the sole "c" item takes slot 1
    slot1 = matplotlib.colors.to_rgb(cac.CATEGORICAL_LIGHT[0])
    assert np.allclose(first[:3], slot1, atol=0.2)


def test_compact_number():
    cases = [((5,), "5"), ((1900,), "1.9K"), ((1900, "", "B"), "1,900B"),
             ((1.9e12,), "1.9T"),
             ((2.5e6,), "2.5M"), ((0.037,), "0.037"),
             ((42, "$", "B"), "$42B"), ((1900e9, "$"), "$1.9T")]
    for args, want in cases:
        got = cac.compact_number(*args)
        assert got == want, f"compact_number{args} == {got!r}, wanted {want!r}"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {test.__name__}: {exc}")
        else:
            print(f"ok   {test.__name__}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
