#!/usr/bin/env python3
"""Bounds on A(n, d) for cubes too big to settle exactly.

`hypercube_packing.py` pins down A(n, d) exactly for n, d <= 9.  Past that this
script reports what it can honestly derive -- a lower bound from a code it
actually builds, an upper bound from a proved theorem, and an exact value only
where the two coincide.

Exactness runs out well before the published record does, and the record itself
runs out: A(15, 3) = 2048 is known, while A(16, 3) is open and has been for
decades.  Use --literature to see the published best-known bounds alongside
these, and --compare to see the difference.

Lower bounds come from
  * the greedy lexicode and the cyclic-orbit search, while those stay affordable,
  * classical linear families: Hamming, extended Hamming, Reed-Muller (which
    covers the simplex and first-order Hadamard codes) and the binary Golay code,
  * Hadamard matrices from the Sylvester and Paley constructions,
  * the Gilbert-Varshamov counting bound, which needs no construction at all,
  * and propagation: padding, shortening, puncturing and parity extension carry
    a good code outwards to all its neighbours in the table.

Upper bounds come from the Singleton, sphere-packing and Plotkin bounds, the
halving bound A(n,d) <= 2A(n-1,d), the puncturing bound A(n,d) <= A(n-1,d-1),
monotonicity in d, and the parity identity A(n,d) = A(n-1,d-1) for even d.

Both families are closed under those relations until nothing moves, which is
how one strong code or one sharp bound spreads across the whole table.

Where they meet the cell is exact and proved here -- that includes every
perfect code (Hamming at n = 7, 15, 31; Golay at n = 23) and the whole
Plotkin-tight ridge around d = n/2 wherever a Hadamard matrix exists.

Usage
-----
    python3 hypercube_bounds.py                       # n <= 24, our own bounds
    python3 hypercube_bounds.py --literature          # published best known
    python3 hypercube_bounds.py --compare             # ours against published
    python3 hypercube_bounds.py --percent --min-d 1   # share of the 2^n vertices
    python3 hypercube_bounds.py --format markdown --stats
"""

from __future__ import annotations

import argparse
import sys
from math import comb

from hypercube_packing import (cyclic_orbit_code, hamming, lexicode,
                               min_distance, popcount, verify)

try:
    import brouwer_table
except ImportError:                              # reference data is optional
    brouwer_table = None

# --------------------------------------------------------------------------
# linear algebra over F2, on integer bitmasks
# --------------------------------------------------------------------------


def span(generators):
    """Every F2-combination of `generators` (returned as a sorted list)."""
    words = [0]
    for g in generators:
        words += [w ^ g for w in words]
    return sorted(set(words))


def min_nonzero_weight(words):
    """Minimum distance of a *linear* code is its minimum nonzero weight."""
    best = None
    for w in words:
        if w:
            c = popcount(w)
            if best is None or c < best:
                best = c
    return best


def rank(vectors):
    basis = []
    for v in vectors:
        for b in basis:
            v = min(v, v ^ b)
        if v:
            basis.append(v)
            basis.sort(reverse=True)
    return len(basis)


# --------------------------------------------------------------------------
# classical linear families
# --------------------------------------------------------------------------


def hamming_code(r):
    """The perfect [2^r - 1, 2^r - 1 - r, 3] Hamming code."""
    n = (1 << r) - 1
    k = n - r
    if n > 20:                      # too wide to enumerate; parameters are theory
        return n, 1 << k, 3, None
    # columns of the parity check matrix are the nonzero r-bit patterns
    words = [v for v in range(1 << n)
             if all(popcount(v & _column_mask(n, bit)) % 2 == 0 for bit in range(r))]
    return n, len(words), min_nonzero_weight(words), words


def _column_mask(n, bit):
    """Positions j (0-based) whose column j+1 has `bit` set."""
    m = 0
    for j in range(n):
        if (j + 1) >> bit & 1:
            m |= 1 << j
    return m


def reed_muller(order, m):
    """RM(order, m): length 2^m, distance 2^(m-order).

    Rows are the evaluation vectors of the monomials of degree <= order in m
    boolean variables.  RM(1, m) is the first-order (Hadamard) code and
    RM(m-1, m) is the even-weight code; the simplex code is RM(1, m) punctured.
    """
    n = 1 << m
    dim = sum(comb(m, i) for i in range(order + 1))
    if dim > 14:                    # 2^dim codewords; anything larger is a
        return n, 0, 0, None        # low-distance code the trivial columns cover
    coords = list(range(n))
    gens = []
    for degree in range(order + 1):
        for subset in _subsets(range(m), degree):
            v = 0
            for x in coords:
                if all((x >> i) & 1 for i in subset):
                    v |= 1 << x
            gens.append(v)
    words = span(gens)
    return n, len(words), min_nonzero_weight(words), words


def _subsets(items, size):
    items = list(items)
    if size == 0:
        yield ()
        return
    for i, it in enumerate(items):
        for rest in _subsets(items[i + 1:], size - 1):
            yield (it,) + rest


def golay23():
    """The perfect binary Golay code [23, 12, 7], as a cyclic code."""
    # x^11 + x^10 + x^6 + x^5 + x^4 + x^2 + 1
    g = (1 << 11) | (1 << 10) | (1 << 6) | (1 << 5) | (1 << 4) | (1 << 2) | 1
    words = span([g << i for i in range(12)])
    return 23, len(words), min_nonzero_weight(words), words


# --------------------------------------------------------------------------
# Hadamard matrices -> codes meeting the Plotkin bound at n = 2d
# --------------------------------------------------------------------------


def _sylvester(order_log):
    """Hadamard matrix of order 2^k as a list of +-1 row tuples."""
    rows = [(1,)]
    for _ in range(order_log):
        rows = ([r + r for r in rows] +
                [r + tuple(-x for x in r) for r in rows])
    return rows


def _paley1(q):
    """Hadamard matrix of order q+1 for an odd prime q = 3 (mod 4)."""
    residues = {(i * i) % q for i in range(1, q)}

    def chi(x):
        x %= q
        if x == 0:
            return 0
        return 1 if x in residues else -1

    # H = [[1, 1...1], [-1, Q + I]] with Q the Jacobsthal matrix Q_ij = chi(i-j).
    # (Q+I)(Q+I)^T = (q+1)I - J and Q has zero row sums, which is what makes
    # every pair of rows orthogonal.
    n = q + 1
    rows = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == 0:
                row.append(1)
            elif j == 0:
                row.append(-1)
            elif i == j:
                row.append(1)
            else:
                row.append(chi(i - j))
        rows.append(tuple(row))
    return rows


def _double(rows):
    """Order N Hadamard matrix -> order 2N, by the Sylvester doubling step."""
    return ([r + r for r in rows] +
            [r + tuple(-x for x in r) for r in rows])


def _is_hadamard(rows):
    n = len(rows)
    if any(len(r) != n for r in rows):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            if sum(a * b for a, b in zip(rows[i], rows[j])) != 0:
                return False
    return True


def hadamard_matrices(limit):
    """Every Hadamard order <= limit we can reach, as +-1 row lists."""
    mats = {}

    def add(rows):
        n = len(rows)
        if 2 <= n <= limit and n not in mats and _is_hadamard(rows):
            mats[n] = rows

    k = 1
    while (1 << k) <= limit:
        add(_sylvester(k))
        k += 1
    for q in range(3, limit):
        if q % 4 == 3 and _is_prime(q):
            add(_paley1(q))

    changed = True
    while changed:                               # doubling reaches 24, 40, ...
        changed = False
        for n in sorted(mats):
            if 2 * n <= limit and 2 * n not in mats:
                add(_double(mats[n]))
                changed = 2 * n in mats
    return mats


def hadamard_codes(limit):
    """(n, size, d, code) from each Hadamard matrix we can build.

    The rows as 0/1 vectors, together with their complements, form a code of
    size 2n and minimum distance n/2 -- exactly the Plotkin bound at n = 2d,
    so these cells come out exact.  The distance is measured, not assumed.
    """
    out = []
    for n, rows in sorted(hadamard_matrices(limit).items()):
        words = []
        for r in rows:
            v = 0
            for j, x in enumerate(r):
                if x < 0:
                    v |= 1 << j
            words.append(v)
        full = sorted(set(words) | {w ^ ((1 << n) - 1) for w in words})
        d = min_distance(full)
        if d is not None and len(full) == 2 * n:
            out.append((n, len(full), d, full))
    return out


def _is_prime(x):
    if x < 2:
        return False
    i = 2
    while i * i <= x:
        if x % i == 0:
            return False
        i += 1
    return True


# --------------------------------------------------------------------------
# bounds that need no construction
# --------------------------------------------------------------------------


def gilbert_varshamov(n, d):
    """A(n, d) >= 2^n / V(n, d-1): greedily fill until every ball is hit."""
    ball = sum(comb(n, i) for i in range(d))
    return -(-(1 << n) // ball)                  # ceiling division


def singleton(n, d):
    return 1 << (n - d + 1)


def sphere_packing(n, d):
    t = (d - 1) // 2
    return (1 << n) // sum(comb(n, i) for i in range(t + 1))


def plotkin(n, d):
    """Plotkin's bound, in both the even-d and odd-d forms."""
    if d % 2 == 0:
        if 2 * d > n:
            return 4 * d if n == 2 * d else 2 * (d // (2 * d - n))
    else:
        if 2 * d + 1 > n:
            return 4 * d + 4 if n == 2 * d + 1 else 2 * ((d + 1) // (2 * d + 1 - n))
    return None


# --------------------------------------------------------------------------
# assembling the table
# --------------------------------------------------------------------------


class Bounds:
    """Lower and upper bound for one cell, each with a one-word provenance."""

    __slots__ = ("lo", "hi", "why_lo", "why_hi")

    def __init__(self, lo, hi, why_lo="", why_hi=""):
        self.lo, self.hi = lo, hi
        self.why_lo, self.why_hi = why_lo, why_hi

    @property
    def exact(self):
        return self.lo == self.hi

    def raise_lo(self, value, why):
        if value > self.lo:
            self.lo, self.why_lo = value, why
            return True
        return False

    def lower_hi(self, value, why):
        if value < self.hi:
            self.hi, self.why_hi = value, why
            return True
        return False


def build(max_n, max_d, lex_limit=14, orbit_limit=10, margin=4, verbose=False):
    """Bounds for every (n, d) with n <= max_n, d <= max_d."""
    top_n = max_n + margin                       # extra rows feed propagation
    top_d = max_d + margin
    T = {(n, d): Bounds(1, 1 << n) for n in range(1, top_n + 1)
         for d in range(1, top_d + 1)}

    def say(msg):
        if verbose:
            print(msg, file=sys.stderr, flush=True)

    # ---- exact base cases -------------------------------------------------
    for n in range(1, top_n + 1):
        for d in range(1, top_d + 1):
            c = T[(n, d)]
            if d > n:
                c.lo = c.hi = 1
                c.why_lo = c.why_hi = "d > n"
            elif d == 1:
                c.lo = c.hi = 1 << n
                c.why_lo = c.why_hi = "all vertices"
            elif d == 2:
                c.lo = c.hi = 1 << (n - 1)
                c.why_lo = c.why_hi = "even-weight half"
            elif d == n:
                c.lo = c.hi = 2
                c.why_lo = c.why_hi = "antipodal pair"

    # ---- closed-form upper bounds ----------------------------------------
    for n in range(1, top_n + 1):
        for d in range(3, min(n, top_d) + 1):
            if d == n:
                continue
            c = T[(n, d)]
            c.lower_hi(singleton(n, d), "Singleton")
            c.lower_hi(sphere_packing(n, d), "sphere-packing")
            pb = plotkin(n, d)
            if pb is not None:
                c.lower_hi(pb, "Plotkin")

    # ---- constructive lower bounds ---------------------------------------
    for n in range(1, top_n + 1):
        for d in range(3, min(n, top_d) + 1):
            T[(n, d)].raise_lo(gilbert_varshamov(n, d), "Gilbert-Varshamov")

    def offer_code(n, size, d, words, label):
        """Register a construction, verifying it whenever that is affordable."""
        if n > top_n or d < 1:
            return
        if words is not None and len(words) <= 4096:
            verify(words, n, d)
        for dd in range(1, min(d, top_d) + 1):   # a d-code is also a dd-code
            if (n, dd) in T:
                T[(n, dd)].raise_lo(size, label)
        say(f"  {label:22s} ({n}, {size}, {d})")

    say("constructions:")
    for r in range(2, 6):
        n, size, d, words = hamming_code(r)
        if n <= top_n:
            offer_code(n, size, d, words, "Hamming")
            if n + 1 <= top_n:                   # extended: [2^r, k, 4]
                offer_code(n + 1, size, 4, None, "extended Hamming")
    for m in range(2, 6):
        if (1 << m) > top_n:
            break
        for order in range(1, m):
            n, size, d, words = reed_muller(order, m)
            if size:
                offer_code(n, size, d, words, f"RM({order},{m})")
    if top_n >= 23:
        n, size, d, words = golay23()
        offer_code(n, size, d, words, "Golay")
        if top_n >= 24:
            offer_code(24, size, 8, None, "extended Golay")
    for n, size, d, words in hadamard_codes(min(top_n, 64)):
        offer_code(n, size, d, words, "Hadamard")

    say("searches:")
    for n in range(1, min(top_n, lex_limit) + 1):
        for d in range(3, min(n, top_d) + 1):
            code = lexicode(n, d)
            T[(n, d)].raise_lo(len(code), "lexicode")
    for n in range(1, min(top_n, orbit_limit) + 1):
        for d in range(3, min(n, top_d) + 1):
            code = cyclic_orbit_code(n, d)
            T[(n, d)].raise_lo(len(code), "cyclic orbits")

    # ---- close under the standard relations ------------------------------
    changed = True
    rounds = 0
    while changed:
        changed = False
        rounds += 1
        for n in range(1, top_n + 1):
            for d in range(1, top_d + 1):
                c = T[(n, d)]

                # ---- lower bounds propagate outwards
                if d + 1 <= top_d:
                    changed |= c.raise_lo(T[(n, d + 1)].lo, "harder d")
                if n > 1:
                    changed |= c.raise_lo(T[(n - 1, d)].lo, "pad a coordinate")
                if n + 1 <= top_n:
                    changed |= c.raise_lo(-(-T[(n + 1, d)].lo // 2), "shorten")
                    if d + 1 <= top_d:
                        changed |= c.raise_lo(T[(n + 1, d + 1)].lo, "puncture")
                if d % 2 == 1 and n + 1 <= top_n and d + 1 <= top_d:
                    changed |= T[(n + 1, d + 1)].raise_lo(c.lo, "parity extension")

                # ---- upper bounds propagate inwards
                if n > 1:
                    changed |= c.lower_hi(2 * T[(n - 1, d)].hi, "halving")
                    if d > 1:
                        changed |= c.lower_hi(T[(n - 1, d - 1)].hi, "puncture")
                if d > 1:
                    changed |= c.lower_hi(T[(n, d - 1)].hi, "easier d")

                # ---- the parity identity is an equality, so it moves both
                if d % 2 == 0 and n > 1 and d > 1:
                    src = T[(n - 1, d - 1)]
                    changed |= c.raise_lo(src.lo, "parity identity")
                    changed |= c.lower_hi(src.hi, "parity identity")
                    changed |= src.raise_lo(c.lo, "parity identity")
                    changed |= src.lower_hi(c.hi, "parity identity")
        if rounds > 200:
            break
    say(f"closure converged after {rounds} rounds")

    for (n, d), c in T.items():
        if c.lo > c.hi:
            raise AssertionError(f"inconsistent bounds at A({n},{d}): "
                                 f"{c.lo} > {c.hi} ({c.why_lo} / {c.why_hi})")
    return {(n, d): T[(n, d)] for n in range(1, max_n + 1)
            for d in range(1, max_d + 1)}


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def literature_table(max_n, max_d):
    """The published best-known bounds, in the same shape as build()."""
    if brouwer_table is None:
        raise SystemExit("brouwer_table.py is not available")
    T = {}
    for n in range(1, max_n + 1):
        for d in range(1, max_d + 1):
            b = brouwer_table.bounds(n, d)
            if b is None:
                T[(n, d)] = Bounds(1, 1 << n, "not tabulated", "not tabulated")
            else:
                T[(n, d)] = Bounds(b[0], b[1], "published", "published")
    return T


def compare(ours, lit, max_n, max_d, min_d):
    """How our from-scratch bounds stack up against the published ones."""
    live = [(n, d) for n in range(1, max_n + 1)
            for d in range(min_d, max_d + 1) if d <= n]
    ours_x = [k for k in live if ours[k].exact]
    lit_x = [k for k in live if lit[k].exact]
    out = ["", f"cells with d <= n:      {len(live)}",
           f"exact, from scratch:    {len(ours_x)}",
           f"exact, published:       {len(lit_x)}"]

    disagree = [k for k in ours_x if not (lit[k].lo == lit[k].hi == ours[k].lo)]
    out.append(f"disagreements:          {len(disagree)}"
               + ("" if not disagree else f"  {disagree}"))

    behind = []
    for k in live:
        if lit[k].lo > ours[k].lo:
            behind.append((lit[k].lo / max(ours[k].lo, 1), k))
    out.append("")
    out.append(f"published lower bound beats ours in {len(behind)} cells; worst:")
    for ratio, (n, d) in sorted(behind, reverse=True)[:6]:
        out.append(f"  A({n:2d},{d:2d})  ours >= {ours[(n, d)].lo:<8}"
                   f" published >= {lit[(n, d)].lo:<8} ({ratio:.1f}x)")
    return "\n".join(out)


def format_percent(fraction):
    """A(n,d)/2^n as a percentage, readable across seven orders of magnitude."""
    pct = 100.0 * fraction
    if pct >= 99.999:
        return "100%"
    if pct >= 0.001:
        return f"{pct:.3g}%"
    return f"{pct:.1e}%"


def cell_text(c, n=None, percent=False):
    """The cell as a count, or as a share of the cube's 2^n vertices."""
    if not percent:
        return str(c.lo) if c.exact else f"{c.lo}-{c.hi}"
    share = format_percent(c.lo / (1 << n))
    return share if c.exact else ">=" + share


def render_table(T, max_n, max_d, min_d, percent=False):
    cols = list(range(min_d, max_d + 1))
    width = max(6, max(len(cell_text(T[(n, d)], n, percent))
                       for n in range(1, max_n + 1) for d in cols) + 2)
    out = (["A(n, d) as a share of the cube's 2^n vertices "
            "(>= means the value is not settled)", ""] if percent else
           ["A(n, d) - exact where the bounds meet, otherwise lower-upper", ""])
    corner = "n \\ d"
    head = f"{corner:>6} |" + "".join(f"{d:>{width}}" for d in cols)
    out.append(head)
    out.append("-" * 6 + "-+" + "-" * (width * len(cols)))
    for n in range(1, max_n + 1):
        row = f"{n:>6} |"
        for d in cols:
            row += f"{cell_text(T[(n, d)], n, percent):>{width}}"
        out.append(row)
    return "\n".join(out)


def render_markdown(T, max_n, max_d, min_d, percent=False):
    cols = list(range(min_d, max_d + 1))
    out = ["| n \\ d | " + " | ".join(str(d) for d in cols) + " |",
           "|---:" * (len(cols) + 1) + "|"]
    for n in range(1, max_n + 1):
        cells = []
        for d in cols:
            c = T[(n, d)]
            text = cell_text(c, n, percent)
            cells.append(f"**{text}**" if c.exact else text)
        out.append(f"| **{n}** | " + " | ".join(cells) + " |")
    out.append("")
    out.append("Bold = exact (lower and upper bound coincide). "
               "Otherwise the cell shows lower-upper.")
    return "\n".join(out)


def render_csv(T, max_n, max_d, min_d):
    cols = list(range(min_d, max_d + 1))
    out = ["n,d,lower,upper,exact,share_of_2^n,why_lower,why_upper"]
    for n in range(1, max_n + 1):
        for d in cols:
            c = T[(n, d)]
            out.append(f"{n},{d},{c.lo},{c.hi},{int(c.exact)},"
                       f"{c.lo / (1 << n):.6g},{c.why_lo},{c.why_hi}")
    return "\n".join(out)


def render_stats(T, max_n, max_d, min_d):
    cols = list(range(min_d, max_d + 1))
    live = [(n, d) for n in range(1, max_n + 1) for d in cols if d <= n]
    exact = [(n, d) for (n, d) in live if T[(n, d)].exact]
    out = ["", f"cells with d <= n:  {len(live)}",
           f"settled exactly:    {len(exact)}  "
           f"({100.0 * len(exact) / max(len(live), 1):.1f}%)"]

    by_reason = {}
    for (n, d) in exact:
        by_reason.setdefault(T[(n, d)].why_hi, []).append((n, d))
    out.append("")
    out.append("exact cells, by the bound that closed them:")
    for why, cells in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
        out.append(f"  {why:22s} {len(cells):4d}")

    widest = sorted(live, key=lambda nd: -(T[nd].hi - T[nd].lo))[:8]
    out.append("")
    out.append("widest remaining gaps:")
    for (n, d) in widest:
        c = T[(n, d)]
        if c.exact:
            continue
        out.append(f"  A({n:2d},{d:2d})  {c.lo:>8} - {c.hi:<8}"
                   f"  lower: {c.why_lo}")
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Lower and upper bounds on A(n, d) for larger cubes.")
    p.add_argument("--max-n", type=int, default=24)
    p.add_argument("--max-d", type=int, default=12)
    p.add_argument("--min-d", type=int, default=3,
                   help="first column shown; d=1,2 are trivially 2^n and 2^(n-1)")
    p.add_argument("--format", choices=("table", "markdown", "csv"), default="table")
    p.add_argument("--stats", action="store_true", help="coverage summary")
    p.add_argument("--literature", action="store_true",
                   help="show the published best-known bounds instead of ours")
    p.add_argument("--compare", action="store_true",
                   help="compare our from-scratch bounds against the published ones")
    p.add_argument("--percent", action="store_true",
                   help="show each value as a share of the cube's 2^n vertices")
    p.add_argument("--lex-limit", type=int, default=14,
                   help="run the lexicode search up to this n (default 14)")
    p.add_argument("--orbit-limit", type=int, default=10,
                   help="run the cyclic-orbit search up to this n (default 10)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    if args.max_n < 1 or args.max_d < 1:
        p.error("--max-n and --max-d must be at least 1")
    if args.min_d < 1 or args.min_d > args.max_d:
        p.error("--min-d must be between 1 and --max-d")

    ours = build(args.max_n, args.max_d, lex_limit=args.lex_limit,
                 orbit_limit=args.orbit_limit, verbose=args.verbose)
    T = literature_table(args.max_n, args.max_d) if args.literature else ours

    if args.compare:
        lit = T if args.literature else literature_table(args.max_n, args.max_d)
        print(compare(ours, lit, args.max_n, args.max_d, args.min_d).lstrip("\n"))
        print()

    if args.format == "markdown":
        print(render_markdown(T, args.max_n, args.max_d, args.min_d, args.percent))
    elif args.format == "csv":
        print(render_csv(T, args.max_n, args.max_d, args.min_d))
    else:
        print(render_table(T, args.max_n, args.max_d, args.min_d, args.percent))
    if args.stats:
        print(render_stats(T, args.max_n, args.max_d, args.min_d))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
