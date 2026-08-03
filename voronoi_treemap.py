"""Area-proportional Voronoi treemap inside a circle (or any convex boundary).

Given a list of positive numbers, this partitions a region into one polygon per
number, where every polygon's area is proportional to its number -- the layout
used by the "polygons inside a circle" infographics.

The algorithm is the additively-weighted (power) Voronoi treemap of
Nocaj & Brandes, "Computing Voronoi Treemaps" (2012):

    repeat until every area matches its target
        1. build the power diagram of the current sites and weights
        2. move each site to the centroid of its cell      (Lloyd relaxation)
        3. Newton step on each weight to correct its area
        4. clamp weights so that no site can swallow another

Only numpy is required; all geometry (half-plane clipping, areas, centroids,
pole of inaccessibility) is implemented here.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "circle_boundary",
    "polygon_area",
    "polygon_centroid",
    "voronoi_treemap",
    "pole_of_inaccessibility",
    "TreemapResult",
]


# --------------------------------------------------------------------------- #
# basic polygon geometry
# --------------------------------------------------------------------------- #

def circle_boundary(center=(0.0, 0.0), radius=1.0, segments=512):
    """A circle approximated as a counter-clockwise polygon."""
    t = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    # the polygon is inscribed, so scale it up to preserve the circle's area
    r = radius / math.sqrt(math.sin(2 * math.pi / segments) * segments / (2 * math.pi))
    return np.column_stack([center[0] + r * np.cos(t), center[1] + r * np.sin(t)])


def polygon_area(poly):
    """Signed area (positive for counter-clockwise vertex order)."""
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def polygon_centroid(poly):
    """Area centroid; falls back to the vertex mean for degenerate polygons."""
    if len(poly) < 3:
        return poly.mean(axis=0) if len(poly) else np.zeros(2)
    x, y = poly[:, 0], poly[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    a = 0.5 * cross.sum()
    if abs(a) < 1e-18:
        return poly.mean(axis=0)
    return np.array([((x + xn) * cross).sum(), ((y + yn) * cross).sum()]) / (6.0 * a)


def _clip_halfplane(verts, owners, normal, offset, owner_id):
    """Sutherland-Hodgman clip of a convex polygon by ``normal . p <= offset``.

    ``owners[k]`` labels the edge ``verts[k] -> verts[k + 1]`` with the index of
    the site that created it (``-1`` for the outer boundary). Keeping track of
    that is what makes the exact Newton step on the weights possible.
    """
    n = len(verts)
    if n == 0:
        return verts, owners
    d = verts @ normal - offset
    inside = d <= 0.0
    if inside.all():
        return verts, owners
    if not inside.any():
        return verts[:0], owners[:0]

    nxt_v = np.roll(verts, -1, axis=0)
    nxt_in = np.roll(inside, -1)
    crossed = inside != nxt_in
    with np.errstate(divide="ignore", invalid="ignore"):
        t = d / (d - np.roll(d, -1))
    cut = verts + np.nan_to_num(t)[:, None] * (nxt_v - verts)

    # Emit, per edge i: the vertex itself if inside, then the crossing point if
    # edge i straddles the line. A crossing that *leaves* the half-plane starts
    # an edge along the cutting line (owner_id); one that enters continues the
    # original edge, so it keeps that edge's owner.
    out_v = np.empty((2 * n, 2))
    out_o = np.empty(2 * n, dtype=np.int64)
    keep = np.empty(2 * n, dtype=bool)
    out_v[0::2], out_v[1::2] = verts, cut
    out_o[0::2], out_o[1::2] = owners, np.where(inside, owner_id, owners)
    keep[0::2], keep[1::2] = inside, crossed
    return out_v[keep], out_o[keep]


def power_cells(sites, weights, boundary):
    """Power (additively weighted Voronoi) diagram clipped to ``boundary``.

    Returns a list of ``(vertices, edge_owners)`` pairs, one per site, in the
    same order as ``sites``. Cells may be empty when a site is dominated.
    """
    sites = np.asarray(sites, dtype=float)
    weights = np.asarray(weights, dtype=float)
    n = len(sites)
    base_owners = np.full(len(boundary), -1, dtype=np.int64)
    sq = (sites ** 2).sum(axis=1)

    # cell_i = { x : |x - p_i|^2 - w_i <= |x - p_j|^2 - w_j  for all j }
    #        = { x : 2 (p_j - p_i).x <= |p_j|^2 - |p_i|^2 - w_j + w_i }
    d2 = ((sites[:, None, :] - sites[None, :, :]) ** 2).sum(axis=-1)
    np.fill_diagonal(d2, np.inf)
    # signed distance from site i to its bisector with j (>= 0 once the weights
    # are clamped), which lets us sort the cuts near-to-far and stop early
    with np.errstate(invalid="ignore"):
        reach = (d2 + weights[:, None] - weights[None, :]) / (2.0 * np.sqrt(d2))
    np.fill_diagonal(reach, np.inf)
    order = np.argsort(reach, axis=1, kind="stable")

    cells = []
    for i in range(n):
        verts, owners = boundary, base_owners
        rmax = float(np.linalg.norm(boundary - sites[i], axis=1).max())
        for j in order[i]:
            if reach[i, j] > rmax:
                break  # this bisector, and every later one, misses the cell
            normal = 2.0 * (sites[j] - sites[i])
            offset = sq[j] - sq[i] - weights[j] + weights[i]
            verts, owners = _clip_halfplane(verts, owners, normal, offset, int(j))
            if len(verts) < 3:
                verts, owners = verts[:0], owners[:0]
                break
            rmax = float(np.linalg.norm(verts - sites[i], axis=1).max())
        cells.append((verts, owners))
    return cells


# --------------------------------------------------------------------------- #
# the treemap itself
# --------------------------------------------------------------------------- #

@dataclass
class TreemapResult:
    """Output of :func:`voronoi_treemap`."""

    cells: list          # one (m, 2) vertex array per input value, input order
    sites: np.ndarray    # (n, 2) final site positions
    weights: np.ndarray  # (n,) final power weights
    areas: np.ndarray    # (n,) achieved cell areas
    targets: np.ndarray  # (n,) requested cell areas
    iterations: int
    max_error: float     # worst relative area error, e.g. 0.004 == 0.4 %
    roundness: float = 1.0  # worst 4*pi*A/P^2 over the cells; 1 = circle

    def __iter__(self):
        return iter(self.cells)

    def __len__(self):
        return len(self.cells)


def _initial_sites(values, boundary, rng):
    """Seed sites on a sunflower spiral, biggest value nearest the middle.

    Even spacing converges much faster than uniform random points, and putting
    the largest values in the centre keeps them from being squeezed against the
    boundary -- which is also what makes the finished diagram look balanced.
    """
    n = len(values)
    centre = polygon_centroid(boundary)
    radius = np.linalg.norm(boundary - centre, axis=1).min()
    golden = math.pi * (3.0 - math.sqrt(5.0))

    k = np.arange(n)
    r = 0.92 * radius * np.sqrt((k + 0.5) / n)
    theta = k * golden + rng.uniform(0, 2 * math.pi)
    spiral = centre + np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    spiral += rng.normal(scale=0.01 * radius, size=spiral.shape)

    sites = np.empty_like(spiral)
    sites[np.argsort(-np.asarray(values, dtype=float), kind="stable")] = spiral
    return sites


def _clamp_weights(sites, weights, slack=0.999):
    """Enforce ``w_i - w_j <= d_ij^2`` so every site keeps a non-empty cell."""
    d2 = ((sites[:, None, :] - sites[None, :, :]) ** 2).sum(axis=-1)
    np.fill_diagonal(d2, np.inf)
    for _ in range(4):
        limit = (weights[None, :] + slack * d2).min(axis=1)
        over = weights > limit
        if not over.any():
            break
        weights = np.where(over, limit, weights)
    return weights


def worst_roundness(cells):
    """Smallest ``4 pi A / P^2`` over the cells: 1 is a circle, 0 is a sliver.

    Every layout that meets the area tolerance is correct; this is what tells
    the good-looking ones apart from the ones with a splinter wedged against
    the rim.
    """
    best = 1.0
    for verts in cells:
        if len(verts) < 3:
            return 0.0
        area = abs(polygon_area(verts))
        perim = np.linalg.norm(np.roll(verts, -1, axis=0) - verts, axis=1).sum()
        if perim <= 0:
            return 0.0
        best = min(best, 4.0 * math.pi * area / perim ** 2)
    return best


def voronoi_treemap(
    values,
    boundary=None,
    *,
    center=(0.0, 0.0),
    radius=1.0,
    max_iter=400,
    tol=0.004,
    damping=0.75,
    seed=0,
    attempts=3,
    verbose=False,
):
    """Lay out one polygon per value, with area proportional to the value.

    Parameters
    ----------
    values : sequence of positive numbers
    boundary : (m, 2) array, optional
        Convex outer polygon. Defaults to a circle at ``center`` / ``radius``.
    max_iter, tol : int, float
        Stop once the worst relative area error drops below ``tol`` (0.004 =
        0.4 %), or after ``max_iter`` sweeps.
    damping : float
        Fraction of the Newton step applied per iteration; lower is slower but
        steadier.
    seed : int
        Controls the initial site placement, so layouts are reproducible.
    attempts : int
        Number of seeds to try (``seed``, ``seed + 1``, ...), keeping the
        best-shaped layout among those that hit the tolerance. Cell shape
        depends on where the sites started, so a couple of extra tries is the
        cheapest way to avoid a splinter-shaped cell. Set to 1 to skip.

    Returns
    -------
    TreemapResult
    """
    best = None
    for k in range(max(int(attempts), 1)):
        run = _layout_once(values, boundary, center=center, radius=radius,
                           max_iter=max_iter, tol=tol, damping=damping,
                           seed=seed + k, verbose=verbose)
        run.roundness = worst_roundness(run.cells)
        if verbose:
            print(f"attempt {k + 1}: error {run.max_error:.3%}, "
                  f"worst roundness {run.roundness:.3f}")
        if best is None:
            best = run
            continue
        # accuracy first, then shape
        if (run.max_error < tol) and (best.max_error >= tol):
            best = run
        elif (run.max_error < tol) == (best.max_error < tol):
            key_new = run.roundness if run.max_error < tol else -run.max_error
            key_old = best.roundness if best.max_error < tol else -best.max_error
            if key_new > key_old:
                best = run
    return best


def _layout_once(values, boundary=None, *, center, radius, max_iter, tol,
                 damping, seed, verbose):
    values = np.asarray(values, dtype=float).ravel()
    if values.size == 0:
        raise ValueError("values must not be empty")
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("all values must be finite and strictly positive")

    if boundary is None:
        boundary = circle_boundary(center, radius)
    boundary = np.asarray(boundary, dtype=float)
    if polygon_area(boundary) < 0:
        boundary = boundary[::-1]

    n = len(values)
    total = abs(polygon_area(boundary))
    targets = values / values.sum() * total

    if n == 1:
        return TreemapResult([boundary], np.array([polygon_centroid(boundary)]),
                             np.zeros(1), np.array([total]), targets, 0, 0.0)

    rng = np.random.default_rng(seed)
    sites = _initial_sites(values, boundary, rng)
    weights = np.zeros(n)
    scale = math.sqrt(total)

    best_cells, best_err, iterations = None, np.inf, 0
    for iterations in range(1, max_iter + 1):
        cells = power_cells(sites, weights, boundary)
        areas = np.array([abs(polygon_area(v)) for v, _ in cells])

        # Revive any site that lost its cell: drop it into the roomiest cell.
        dead = np.flatnonzero(areas <= 0.0)
        if dead.size:
            for i in dead:
                host = int(np.argmax(areas))
                sites[i] = pole_of_inaccessibility(cells[host][0])[0] \
                    if len(cells[host][0]) >= 3 else polygon_centroid(boundary)
                sites[i] += rng.normal(scale=0.01 * scale, size=2)
                weights[i] = weights.mean()
            weights = _clamp_weights(sites, weights)
            continue

        err = float(np.max(np.abs(areas - targets) / targets))
        if err < best_err:
            best_err, best_cells = err, [v for v, _ in cells]
        if verbose and iterations % 25 == 0:
            print(f"  iter {iterations:4d}   worst area error {err:7.3%}")
        if err < tol:
            break

        # 1. Lloyd relaxation -- rounder, more "organic" cells.
        for i, (verts, _) in enumerate(cells):
            sites[i] = polygon_centroid(verts)

        # 2. Newton step on the weights. Raising w_i by dw pushes every shared
        #    edge outwards by dw / (2 d_ij), so dA_i = dw * sum(len_ij / 2 d_ij).
        dist = np.linalg.norm(sites[:, None, :] - sites[None, :, :], axis=-1)
        np.fill_diagonal(dist, np.inf)
        gain = np.zeros(n)
        for i, (verts, owners) in enumerate(cells):
            seg = np.linalg.norm(np.roll(verts, -1, axis=0) - verts, axis=1)
            shared = owners >= 0
            if shared.any():
                gain[i] = (seg[shared] / (2.0 * dist[i, owners[shared]])).sum()
        step = np.where(gain > 0, (targets - areas) / np.maximum(gain, 1e-12), 0.0)

        cap = 0.5 * dist.min(axis=1) ** 2
        weights += damping * np.clip(step, -cap, cap)
        weights -= weights.mean()
        weights = _clamp_weights(sites, weights)
    else:
        cells = power_cells(sites, weights, boundary)
        areas = np.array([abs(polygon_area(v)) for v, _ in cells])
        err = float(np.max(np.abs(areas - targets) / targets))
        if err < best_err:
            best_err, best_cells = err, [v for v, _ in cells]

    areas = np.array([abs(polygon_area(v)) for v in best_cells])
    return TreemapResult(best_cells, sites, weights, areas, targets, iterations, best_err)


# --------------------------------------------------------------------------- #
# label placement
# --------------------------------------------------------------------------- #

def _signed_distance(point, poly):
    """Distance from ``point`` to the polygon outline, positive inside."""
    a = poly
    b = np.roll(poly, -1, axis=0)
    ab = b - a
    ap = point - a
    t = np.clip((ap * ab).sum(axis=1) / np.maximum((ab * ab).sum(axis=1), 1e-30), 0.0, 1.0)
    d = np.linalg.norm(ap - t[:, None] * ab, axis=1).min()

    x, y = point
    cross = ((a[:, 1] > y) != (b[:, 1] > y))
    with np.errstate(divide="ignore", invalid="ignore"):
        xs = a[:, 0] + (y - a[:, 1]) / np.where(ab[:, 1] == 0, np.nan, ab[:, 1]) * ab[:, 0]
    inside = np.count_nonzero(cross & (x < xs)) % 2 == 1
    return d if inside else -d


def pole_of_inaccessibility(poly, precision_ratio=0.01):
    """Centre and radius of the largest circle that fits inside ``poly``.

    Mapbox's "polylabel" quadtree search. This is where a label belongs: unlike
    the centroid it is always inside the polygon, and it tells you exactly how
    much room the text has.
    """
    poly = np.asarray(poly, dtype=float)
    if len(poly) < 3:
        return (poly.mean(axis=0) if len(poly) else np.zeros(2)), 0.0

    lo, hi = poly.min(axis=0), poly.max(axis=0)
    size = hi - lo
    cell = float(min(size))
    if cell <= 0:
        return poly.mean(axis=0), 0.0
    precision = max(cell * precision_ratio, 1e-12)

    def make(cx, cy, h):
        d = _signed_distance(np.array([cx, cy]), poly)
        return (-(d + h * math.sqrt(2)), next(counter), cx, cy, h, d)

    counter = iter(range(1 << 30))
    h = cell / 2.0
    queue = []
    x = lo[0] + h
    while x < hi[0] + h:
        y = lo[1] + h
        while y < hi[1] + h:
            heapq.heappush(queue, make(x, y, h))
            y += cell
        x += cell

    bx, by = polygon_centroid(poly)
    best = _signed_distance(np.array([bx, by]), poly)
    while queue:
        neg_bound, _, cx, cy, h, d = heapq.heappop(queue)
        if d > best:
            best, bx, by = d, cx, cy
        if -neg_bound - best <= precision:
            continue
        h /= 2.0
        for sx in (-h, h):
            for sy in (-h, h):
                heapq.heappush(queue, make(cx + sx, cy + sy, h))
    return np.array([bx, by]), max(best, 0.0)
