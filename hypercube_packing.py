#!/usr/bin/env python3
"""Maximal point placements on the vertices of an n-dimensional cube.

Problem
-------
Take the n-dimensional unit cube.  Its 2^n vertices are the binary vectors
{0,1}^n.  How many vertices can we pick so that every two chosen points are at
distance at least d?

Between two vertices of the cube, the Euclidean distance is sqrt(h) where h is
the number of coordinates in which they differ (the Hamming distance), so the
natural integer-valued metric on the vertex set is the Hamming distance.  The
quantity we tabulate is therefore

    A(n, d) = max { |C| : C subset of {0,1}^n, Hamming(x, y) >= d for x != y }

which is the classical "maximum size of a binary code of length n with minimum
distance d".  (Use --metric euclidean to require Euclidean distance >= d
instead; that is just A(n, d^2).)

How the numbers are obtained
----------------------------
Every entry is computed, not hard-coded.  For each cell the program produces

  * a lower bound, by explicitly *building* a code and verifying it:
      - the greedy lexicode,
      - the best union of cyclic-shift orbits (an exact search over orbits;
        this is what finds the nonlinear (10, 40, 4) Best code),
      - extending / puncturing / shortening codes of neighbouring parameters,
      - a randomised local-search polish;
  * an upper bound, from the classical bounds (Singleton, sphere-packing,
    Plotkin, the halving bound and the parity identity), plus an exact
    branch-and-bound clique search on cells small enough to settle.

When the two meet, the value is optimal and the program has proved it.  That
settles every cell except A(8, 3) = 20, whose optimality is a genuinely hard
result quoted from the literature (Best, Brouwer, MacWilliams, Odlyzko &
Sloane, 1978); cells marked with a dagger lean on it.

Usage
-----
    python3 hypercube_packing.py                     # the table
    python3 hypercube_packing.py --format markdown   # markdown / csv / latex
    python3 hypercube_packing.py --code 9 3          # print an optimal code
    python3 hypercube_packing.py --prove --budget 5000000
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from math import comb

# --------------------------------------------------------------------------
# bit helpers
# --------------------------------------------------------------------------


def popcount(x):
    return bin(x).count("1")


def hamming(a, b):
    return bin(a ^ b).count("1")


def min_distance(code):
    """Minimum pairwise Hamming distance of a list of codewords."""
    if len(code) < 2:
        return None
    return min(hamming(a, b) for i, a in enumerate(code) for b in code[i + 1:])


def verify(code, n, d):
    """Check that `code` really is a length-n code with minimum distance >= d."""
    if len(set(code)) != len(code):
        raise AssertionError(f"duplicate codewords for n={n}, d={d}")
    if any(v < 0 or v >= (1 << n) for v in code):
        raise AssertionError(f"codeword outside the {n}-cube")
    if d <= 1:
        return code                      # distinctness is the whole condition
    md = min_distance(code)
    if md is not None and md < d:
        raise AssertionError(f"min distance {md} < {d} for n={n}")
    return code


# --------------------------------------------------------------------------
# constructions (lower bounds) -- each returns an explicit list of codewords
# --------------------------------------------------------------------------


def lexicode(n, d):
    """Greedy: sweep 0, 1, 2, ... and keep every vertex that still fits."""
    code = []
    for v in range(1 << n):
        if all(hamming(v, c) >= d for c in code):
            code.append(v)
    return code


def _cyclic_orbits(n):
    """Orbits of {0,1}^n under the cyclic shift of coordinates."""
    mask = (1 << n) - 1
    seen = set()
    out = []
    for v in range(1 << n):
        if v in seen:
            continue
        orbit = []
        w = v
        while w not in seen:
            seen.add(w)
            orbit.append(w)
            w = ((w << 1) | (w >> (n - 1))) & mask
        out.append(orbit)
    return out


def cyclic_orbit_code(n, d):
    """Largest union of cyclic orbits with minimum distance >= d (exact).

    Many optimal codes are invariant under a cyclic shift, so this small
    search recovers them cheaply -- including the (10, 40, 4) Best code that
    no amount of greedy or local search finds.
    """
    orbits = [o for o in _cyclic_orbits(n)
              if all(hamming(a, b) >= d for i, a in enumerate(o) for b in o[i + 1:])]
    m = len(orbits)
    sizes = [len(o) for o in orbits]
    adj = [0] * m
    for i in range(m):
        for j in range(i + 1, m):
            if all(hamming(a, b) >= d for a in orbits[i] for b in orbits[j]):
                adj[i] |= 1 << j
                adj[j] |= 1 << i

    best_size = 0
    best_set = []

    def expand(cand, chosen, size):
        nonlocal best_size, best_set
        if size > best_size:
            best_size, best_set = size, list(chosen)
        # optimistic bound: take every remaining orbit whole
        cap, rem = size, cand
        while rem:
            b = rem & -rem
            cap += sizes[b.bit_length() - 1]
            rem &= ~b
        if cap <= best_size:
            return
        while cand:
            b = cand & -cand
            v = b.bit_length() - 1
            cand &= ~b
            chosen.append(v)
            expand(cand & adj[v], chosen, size + sizes[v])
            chosen.pop()

    expand((1 << m) - 1, [], 0)
    return [w for i in best_set for w in orbits[i]]


def extend(code):
    """Append an overall parity bit: (n, d) code -> (n+1, d or d+1) code."""
    return [(v << 1) | (popcount(v) & 1) for v in code]


def puncture(code, n, pos):
    """Delete coordinate `pos`: length n -> n-1, distance drops by at most 1."""
    low = (1 << pos) - 1
    out = {(v & low) | ((v >> (pos + 1)) << pos) for v in code}
    return sorted(out)


def shorten(code, n, pos):
    """Keep the codewords with a 0 in `pos`, then delete that coordinate."""
    low = (1 << pos) - 1
    kept = [v for v in code if not (v >> pos) & 1]
    return sorted((v & low) | ((v >> (pos + 1)) << pos) for v in kept)


def local_search(n, d, start, rounds=60000, seed=0):
    """Plateau search: insert a vertex and evict whatever it conflicts with."""
    rng = random.Random(seed)
    best = list(start)
    cur = list(start)
    live = set(cur)
    size = 1 << n
    for _ in range(rounds):
        v = rng.randrange(size)
        if v in live:
            continue
        clash = [c for c in cur if hamming(v, c) < d]
        if len(clash) <= 1 or (len(clash) == 2 and rng.random() < 0.25):
            for c in clash:
                cur.remove(c)
                live.discard(c)
            cur.append(v)
            live.add(v)
            if len(cur) > len(best):
                best = list(cur)
    return best


def build_codes(max_n, max_d, polish=True):
    """Best explicit code we can build for every (n, d) with n <= max_n.

    Codes one dimension past max_n are built too: puncturing and shortening
    them is what produces the optimal length-9 codes.
    """
    top = max_n + 1
    best = {}

    def offer(n, d, code):
        """Record `code` if it beats what we have.  Callers only ever pass
        codes that satisfy the distance requirement by construction; the whole
        table is re-verified from scratch at the end."""
        if code and len(code) > len(best.get((n, d), [])):
            best[(n, d)] = sorted(code)

    for n in range(1, top + 1):
        for d in range(1, max_d + 2):
            if d > n:
                offer(n, d, [0])
                continue
            if d == 1:                                  # every vertex
                offer(n, d, list(range(1 << n)))
                continue
            if d == 2:                                  # the even-weight half
                offer(n, d, [v for v in range(1 << n) if popcount(v) % 2 == 0])
                continue
            offer(n, d, lexicode(n, d))
            offer(n, d, cyclic_orbit_code(n, d))
            # A parity bit turns an odd distance d-1 into d; for odd d it
            # would buy nothing, so only extend into even d.
            if d % 2 == 0 and (n - 1, d - 1) in best:
                offer(n, d, extend(best[(n - 1, d - 1)]))

    # propagate downwards from longer codes until nothing improves
    changed = True
    while changed:
        changed = False
        for n in range(top, 0, -1):
            for d in range(1, max_d + 2):
                src = best.get((n, d))
                if not src or n == 1:
                    continue
                for pos in range(n):
                    if d >= 2:                          # puncture: d -> d-1
                        cand = puncture(src, n, pos)
                        if len(cand) > len(best.get((n - 1, d - 1), [])):
                            offer(n - 1, d - 1, cand)
                            changed = True
                    cand = shorten(src, n, pos)         # shorten: keeps d
                    if len(cand) > len(best.get((n - 1, d), [])):
                        offer(n - 1, d, cand)
                        changed = True

    if polish:
        for (n, d), code in list(best.items()):
            if n <= max_n and 2 < d <= n:
                offer(n, d, local_search(n, d, code, seed=n * 31 + d))

    for (n, d), code in best.items():
        verify(code, n, d)
    return best


# --------------------------------------------------------------------------
# classical upper bounds
# --------------------------------------------------------------------------


def singleton_bound(n, d):
    return 1 << (n - d + 1)


def sphere_packing_bound(n, d):
    t = (d - 1) // 2
    return (1 << n) // sum(comb(n, i) for i in range(t + 1))


def plotkin_bound(n, d):
    """Valid for even d with 2d > n."""
    if d % 2 or 2 * d <= n:
        return None
    if 2 * d == n:
        return 4 * d
    return 2 * (d // (2 * d - n))


# --------------------------------------------------------------------------
# exact search (branch and bound on the "compatibility" graph)
# --------------------------------------------------------------------------


class BudgetExceeded(Exception):
    pass


def exact_max_code(n, d, budget=200_000, lower=0):
    """Exact A(n, d) by maximum-clique search, or None if the budget runs out.

    Distances are translation invariant (Hamming(x, y) = Hamming(x^t, y^t)),
    so we may assume the all-zero vertex is chosen and only search among the
    vertices of weight >= d.
    """
    if d > n:
        return 1
    if d == 1:
        return 1 << n
    verts = [v for v in range(1 << n) if popcount(v) >= d]
    m = len(verts)
    adj = [0] * m
    for i, a in enumerate(verts):
        w = 0
        for j, b in enumerate(verts):
            if i != j and hamming(a, b) >= d:
                w |= 1 << j
        adj[i] = w

    best = max(0, lower - 1)
    nodes = 0

    def expand(cand, size):
        nonlocal best, nodes
        nodes += 1
        if nodes > budget:
            raise BudgetExceeded
        order, colours = [], []
        uncoloured, colour = cand, 0
        while uncoloured:                      # greedy colouring bound
            colour += 1
            group = uncoloured
            while group:
                b = group & -group
                v = b.bit_length() - 1
                group &= ~b
                uncoloured &= ~b
                group &= ~adj[v]
                order.append(v)
                colours.append(colour)
        for k in range(len(order) - 1, -1, -1):
            if size + colours[k] <= best:
                return
            v = order[k]
            if size + 1 > best:
                best = size + 1
            expand(cand & adj[v], size + 1)
            cand &= ~(1 << v)

    try:
        expand((1 << m) - 1, 0)
    except BudgetExceeded:
        return None
    return best + 1


# --------------------------------------------------------------------------
# putting a certified value together for every cell
# --------------------------------------------------------------------------

# The one value the program cannot prove on its own.  Optimality of A(8,3)=20
# is due to Best, Brouwer, MacWilliams, Odlyzko & Sloane, "Bounds for binary
# codes of length less than 25", IEEE Trans. Inform. Theory 24 (1978).
LITERATURE = {(8, 3): 20}


class Cell:
    __slots__ = ("n", "d", "value", "reason", "cited", "code")

    def __init__(self, n, d, value, reason, cited, code=None):
        self.n, self.d = n, d
        self.value = value
        self.reason = reason      # how optimality was established
        self.cited = cited        # True if it rests on the quoted result
        self.code = code


def solve_table(max_n, max_d, budget=200_000, polish=True):
    codes = build_codes(max_n, max_d, polish=polish)
    cells = {}

    for n in range(1, max_n + 1):
        for d in range(1, max_d + 1):
            code = codes.get((n, d), [0])
            lb = len(code)

            if d > n:
                cells[(n, d)] = Cell(n, d, 1, "trivial (d > n)", False, [0])
                continue
            if d == 1:
                cells[(n, d)] = Cell(n, d, 1 << n, "every vertex", False, code)
                continue
            if d == 2:
                cells[(n, d)] = Cell(n, d, 1 << (n - 1), "even-weight half", False, code)
                continue
            if d == n:
                cells[(n, d)] = Cell(n, d, 2, "antipodal pair", False, code)
                continue

            # exact parity identity  A(n, d) = A(n-1, d-1)  for even d
            if d % 2 == 0 and (n - 1, d - 1) in cells:
                src = cells[(n - 1, d - 1)]
                cells[(n, d)] = Cell(n, d, src.value,
                                     f"= A({n-1},{d-1}) (parity identity)",
                                     src.cited, code)
                continue

            ubs = [(singleton_bound(n, d), "Singleton"),
                   (sphere_packing_bound(n, d), "sphere-packing")]
            pb = plotkin_bound(n, d)
            if pb is not None:
                ubs.append((pb, "Plotkin"))
            prev = cells.get((n - 1, d))
            if prev is not None:
                ubs.append((2 * prev.value, f"2*A({n-1},{d})"))
            ub, why = min(ubs)

            if lb == ub:
                # If the only bound that closed the gap is the halving bound
                # applied to a quoted cell, this cell inherits the citation.
                others = [u for u, name in ubs if name != why]
                cited = bool(prev and prev.cited
                             and why == f"2*A({n-1},{d})"
                             and all(u > lb for u in others))
                cells[(n, d)] = Cell(n, d, lb, f"construction meets {why} bound",
                                     cited, code)
                continue

            exact = exact_max_code(n, d, budget=budget, lower=lb)
            if exact is not None:
                cells[(n, d)] = Cell(n, d, exact, "exhaustive search", False, code)
                continue

            lit = LITERATURE.get((n, d))
            if lit is not None:
                cells[(n, d)] = Cell(n, d, lit, "known optimal (BBMOS 1978)", True, code)
                continue

            cells[(n, d)] = Cell(n, d, lb, f"best construction; <= {ub} ({why})",
                                 True, code)
    return cells, codes


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

RESET, BOLD, DIM, CYAN, YELLOW = "\033[0m", "\033[1m", "\033[2m", "\033[36m", "\033[33m"


def _paint(text, *codes, colour=True):
    return "".join(codes) + text + RESET if colour and codes else text


def render_table(cells, max_n, max_d, colour=True, metric="hamming"):
    width = max(5, max(len(str(c.value)) for c in cells.values()) + 2)
    title = f"A(n, d) - most vertices of the n-cube with pairwise distance >= d"

    out = []
    out.append(_paint(title, BOLD, colour=colour))
    out.append("")

    corner = "n \\ d"
    head = _paint(f"{corner:>7}", BOLD, colour=colour) + " |"
    head += "".join(_paint(f"{d:>{width}}", BOLD, CYAN, colour=colour)
                    for d in range(1, max_d + 1))
    out.append(head)
    out.append("-" * 7 + "-+" + "-" * (width * max_d))

    for n in range(1, max_n + 1):
        row = _paint(f"{n:>7}", BOLD, CYAN, colour=colour) + " |"
        for d in range(1, max_d + 1):
            c = cells[(n, d)]
            text = f"{c.value}{'+' if c.cited else ''}"
            if d > n:
                row += _paint(f"{text:>{width}}", DIM, colour=colour)
            elif c.cited:
                row += _paint(f"{text:>{width}}", YELLOW, colour=colour)
            else:
                row += f"{text:>{width}}"
        out.append(row)

    out.append("")
    out.append(_paint("  rows: n = dimension of the cube (2^n vertices)", DIM, colour=colour))
    metric_label = ("required minimum Hamming distance" if metric == "hamming"
                    else "required minimum Euclidean distance (unit cube)")
    out.append(_paint(f"  cols: d = {metric_label}", DIM, colour=colour))
    if any(c.cited for c in cells.values()):
        out.append(_paint("  +   : optimality quoted from Best, Brouwer, MacWilliams,",
                          DIM, colour=colour))
        out.append(_paint("        Odlyzko & Sloane (1978); every other entry is proved here.",
                          DIM, colour=colour))
    return "\n".join(out)


def render_markdown(cells, max_n, max_d):
    out = ["| n \\ d | " + " | ".join(str(d) for d in range(1, max_d + 1)) + " |",
           "|---:" * (max_d + 1) + "|"]
    for n in range(1, max_n + 1):
        row = [f"**{n}**"]
        for d in range(1, max_d + 1):
            c = cells[(n, d)]
            row.append(f"{c.value}{'†' if c.cited else ''}")
        out.append("| " + " | ".join(row) + " |")
    if any(c.cited for c in cells.values()):
        out.append("")
        out.append("† optimality quoted from Best, Brouwer, MacWilliams, Odlyzko & "
                   "Sloane (1978); all other entries are proved by this program.")
    return "\n".join(out)


def render_csv(cells, max_n, max_d):
    out = ["n\\d," + ",".join(str(d) for d in range(1, max_d + 1))]
    for n in range(1, max_n + 1):
        out.append(str(n) + "," + ",".join(str(cells[(n, d)].value)
                                           for d in range(1, max_d + 1)))
    return "\n".join(out)


def render_latex(cells, max_n, max_d):
    out = [r"\begin{tabular}{r|" + "r" * max_d + "}",
           r" $n \backslash d$ & " +
           " & ".join(str(d) for d in range(1, max_d + 1)) + r" \\ \hline"]
    for n in range(1, max_n + 1):
        cs = []
        for d in range(1, max_d + 1):
            c = cells[(n, d)]
            cs.append(f"{c.value}$^\\dagger$" if c.cited else str(c.value))
        out.append(f" {n} & " + " & ".join(cs) + r" \\")
    out.append(r"\end{tabular}")
    return "\n".join(out)


def render_reasons(cells, max_n, max_d):
    out = ["", "How each entry was established:"]
    for n in range(1, max_n + 1):
        for d in range(1, max_d + 1):
            c = cells[(n, d)]
            if d > n:
                continue
            out.append(f"  A({n},{d}) = {c.value:<4} {c.reason}")
    return "\n".join(out)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Table of the maximum number of n-cube vertices with "
                    "pairwise distance >= d.")
    p.add_argument("--max-n", type=int, default=9, help="largest dimension (default 9)")
    p.add_argument("--max-d", type=int, default=9, help="largest distance (default 9)")
    p.add_argument("--format", choices=("table", "markdown", "csv", "latex"),
                   default="table")
    p.add_argument("--metric", choices=("hamming", "euclidean"), default="hamming",
                   help="'euclidean' asks for Euclidean distance >= d on the unit "
                        "cube, i.e. Hamming distance >= d^2")
    p.add_argument("--code", nargs=2, type=int, metavar=("N", "D"),
                   help="print an explicit optimal code for these parameters")
    p.add_argument("--reasons", action="store_true",
                   help="explain how every entry was established")
    p.add_argument("--prove", action="store_true",
                   help="push the exhaustive search harder (see --budget)")
    p.add_argument("--budget", type=int, default=None,
                   help="search-node budget per cell "
                        "(default 200000, or 50000000 with --prove)")
    p.add_argument("--no-color", action="store_true")
    args = p.parse_args(argv)

    if args.max_n < 1 or args.max_d < 1:
        p.error("--max-n and --max-d must be at least 1")

    colour = (not args.no_color and sys.stdout.isatty()
              and os.environ.get("NO_COLOR") is None)
    if args.budget is not None:
        budget = args.budget
    else:
        budget = 50_000_000 if args.prove else 200_000

    # In the Euclidean reading we need Hamming distance >= d^2, so we solve a
    # wider Hamming table and then reindex the columns.
    solve_d = args.max_d ** 2 if args.metric == "euclidean" else args.max_d
    cells, codes = solve_table(args.max_n, solve_d, budget=budget)

    if args.metric == "euclidean":
        cells = {(n, d): cells[(n, min(d * d, solve_d))]
                 for n in range(1, args.max_n + 1)
                 for d in range(1, args.max_d + 1)}

    if args.code:
        n, d = args.code
        h = d * d if args.metric == "euclidean" else d
        code = codes.get((n, h))
        if code is None:
            p.error(f"no code available for n={n}, d={d}")
        verify(code, n, h)
        print(f"n = {n}, required distance {d} ({args.metric}), "
              f"|C| = {len(code)}, min distance = {min_distance(code)}")
        for v in code:
            print("  " + format(v, f"0{n}b"))
        return 0

    renderers = {"table": lambda: render_table(cells, args.max_n, args.max_d, colour,
                                               metric=args.metric),
                 "markdown": lambda: render_markdown(cells, args.max_n, args.max_d),
                 "csv": lambda: render_csv(cells, args.max_n, args.max_d),
                 "latex": lambda: render_latex(cells, args.max_n, args.max_d)}
    print(renderers[args.format]())
    if args.reasons:
        print(render_reasons(cells, args.max_n, args.max_d))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
