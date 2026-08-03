"""Draw a list of numbers as area-proportional polygons packed into a circle.

    from circle_area_chart import circle_area_chart
    fig, ax = circle_area_chart([5, 3, 2, 8, 1])
    fig.savefig("chart.png")

Every polygon's area is exactly proportional to its number (see
``voronoi_treemap.py`` for the layout algorithm); this module is only about
making the result readable and good-looking.

Colour, in the three modes:

* ``color_by="value"`` (default) - one hue, light to dark, a plain sequential
  ramp. It is a redundant encoding of the area, which is what makes the sizes
  comparable across cells that are nowhere near each other.
* ``color_by="group"`` - automatic when ``groups`` is given. Categorical hues in
  fixed slot order (at most 8, the rest fold into "Other"), tinted within each
  group so neighbours stay apart. This is the mode that matches the
  region-coloured infographics.
* ``color_by="single"`` - one flat colour for everything.

Palette, ramps and surfaces come from a validated design system; the
categorical order and the sequential hue are used exactly as documented.

Run ``python circle_area_chart.py --demo`` for an example.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys

import matplotlib
import numpy as np

from voronoi_treemap import pole_of_inaccessibility, voronoi_treemap

__all__ = ["circle_area_chart", "THEMES"]


# --------------------------------------------------------------------------- #
# design tokens
# --------------------------------------------------------------------------- #

# Sequential ramp: one hue, light -> dark. The light end stops at the step that
# still clears the surface, so the smallest cell never dissolves into the page.
BLUE_RAMP_LIGHT = ["#86b6ef", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
                   "#1c5cab", "#184f95", "#104281", "#0d366b"]
# On the dark surface the ramp runs the other way: bigger = brighter = nearer.
BLUE_RAMP_DARK = ["#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5",
                  "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4"]

# Categorical slots, in the fixed validated order. Never cycled: a 9th group is
# folded into "Other".
CATEGORICAL_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                     "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
CATEGORICAL_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500",
                    "#d55181", "#008300", "#9085e9", "#e66767"]

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text_primary": "#0b0b0b",
        "text_secondary": "#52514e",
        "text_muted": "#898781",
        "border": (0.043, 0.043, 0.043, 0.10),
        "ramp": BLUE_RAMP_LIGHT,
        "categorical": CATEGORICAL_LIGHT,
        "other": "#898781",
        "on_dark": "#ffffff",
        "on_light": "#0b0b0b",
    },
    "dark": {
        "surface": "#1a1a19",
        "text_primary": "#ffffff",
        "text_secondary": "#c3c2b7",
        "text_muted": "#898781",
        "border": (1.0, 1.0, 1.0, 0.10),
        "ramp": BLUE_RAMP_DARK,
        "categorical": CATEGORICAL_DARK,
        "other": "#898781",
        "on_dark": "#ffffff",
        "on_light": "#0b0b0b",
    },
}

# Resolved once and pinned on every text object rather than set through
# rcParams: labels are measured before the figure is saved, and an rcParams
# change between the two swaps the font under them -- a different font means
# different metrics, which means every label overflows its cell.
FONT_STACK = ["Helvetica Neue", "Helvetica", "Arial", "Liberation Sans",
              "DejaVu Sans"]
_FONT = None


def font_family():
    """First font of :data:`FONT_STACK` that is actually installed."""
    global _FONT
    if _FONT is None:
        from matplotlib import font_manager
        installed = {f.name for f in font_manager.fontManager.ttflist}
        _FONT = next((f for f in FONT_STACK if f in installed), "DejaVu Sans")
    return _FONT


# --------------------------------------------------------------------------- #
# small colour helpers
# --------------------------------------------------------------------------- #

def _to_rgb(color):
    return np.array(matplotlib.colors.to_rgb(color), dtype=float)


def _hex(rgb):
    return matplotlib.colors.to_hex(np.clip(rgb, 0.0, 1.0))


def _ramp_color(ramp, t):
    """Sample a ramp at ``t`` in [0, 1], interpolating between its steps."""
    t = float(np.clip(t, 0.0, 1.0)) * (len(ramp) - 1)
    lo = int(math.floor(t))
    hi = min(lo + 1, len(ramp) - 1)
    return _hex(_to_rgb(ramp[lo]) * (1 - (t - lo)) + _to_rgb(ramp[hi]) * (t - lo))


def _tint(color, amount):
    """Mix towards white (amount > 0) or black (amount < 0)."""
    target = 1.0 if amount > 0 else 0.0
    return _hex(_to_rgb(color) * (1 - abs(amount)) + target * abs(amount))


def _relative_luminance(color):
    c = _to_rgb(color)
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return float(np.dot(c, [0.2126, 0.7152, 0.0722]))


def _ink_on(color, theme):
    """Pick the label colour with the better contrast against ``color``."""
    lum = _relative_luminance(color)
    on_light = (_relative_luminance(theme["on_light"]) + 0.05)
    on_dark = (_relative_luminance(theme["on_dark"]) + 0.05)
    dark_ratio = (lum + 0.05) / on_light
    light_ratio = on_dark / (lum + 0.05)
    return theme["on_light"] if dark_ratio >= light_ratio else theme["on_dark"]


# --------------------------------------------------------------------------- #
# value formatting
# --------------------------------------------------------------------------- #

def compact_number(v, prefix="", suffix=""):
    """1_900_000 -> '1.9M', 0.037 -> '0.037', 5 -> '5'.

    A caller-supplied ``suffix`` is taken as the unit the numbers are already
    in ("42" + "B"), so the K/M/B/T scaling steps aside for it.
    """
    a = abs(v)
    if not suffix:
        for limit, unit in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
            if a >= limit:
                scaled = v / limit
                text = f"{scaled:.1f}".rstrip("0").rstrip(".") if abs(scaled) < 100 \
                    else f"{scaled:.0f}"
                return f"{prefix}{text}{unit}"
    if a >= 100 or float(v).is_integer():
        return f"{prefix}{v:,.0f}{suffix}"
    return f"{prefix}{v:.3g}{suffix}"


# --------------------------------------------------------------------------- #
# label fitting
# --------------------------------------------------------------------------- #

# Big enough that hinting at small pixel sizes does not skew the measurement:
# a label is measured once at this size and scaled, so a 3 % error here is a
# 3 % overflow on the page.
_PROBE_PT = 36.0
_FIT_MARGIN = 0.97


def _wrap(text, width):
    """Soft-wrap a comma-separated run so a long footnote stays on the page."""
    lines, line = [], ""
    for part in text.split(", "):
        candidate = f"{line}, {part}" if line else part
        if len(candidate) > width and line:
            lines.append(line + ",")
            line = part
        else:
            line = candidate
    lines.append(line)
    return "\n".join(lines)


def _text_size_px(text_obj, renderer):
    bb = text_obj.get_window_extent(renderer=renderer)
    return bb.width, bb.height


def _box_inside(poly, cx, cy, w, h):
    """Is the axis-aligned box fully inside the (convex) polygon?"""
    xs = (cx - w / 2, cx, cx + w / 2)
    ys = (cy - h / 2, cy, cy + h / 2)
    pts = np.array([(x, y) for x in xs for y in ys], dtype=float)
    a = poly
    b = np.roll(poly, -1, axis=0)
    # convex and counter-clockwise: inside means left of every edge
    cross = ((b[:, 0] - a[:, 0])[None, :] * (pts[:, 1, None] - a[None, :, 1])
             - (b[:, 1] - a[:, 1])[None, :] * (pts[:, 0, None] - a[None, :, 0]))
    return bool((cross >= 0).all())


def _fit_scale(poly_px, center_px, w, h, r_px):
    """Largest scale at which a ``w`` x ``h`` text block still fits the cell.

    Start from the exact fit inside the largest empty circle
    (``(sw/2)^2 + (sh/2)^2 = r^2``), then grow into the rest of the cell --
    which is what lets a wide label use a wide cell instead of being sized for
    the circle that fits inside it.
    """
    diag = math.hypot(w, h)
    if diag <= 0:
        return 0.0
    lo = 2.0 * r_px / diag
    if not _box_inside(poly_px, center_px[0], center_px[1], lo * w, lo * h):
        return lo  # anchor sits in a corner; the circle fit is the safe answer
    hi = lo * 4.0
    for _ in range(14):
        mid = (lo + hi) / 2.0
        if _box_inside(poly_px, center_px[0], center_px[1], mid * w, mid * h):
            lo = mid
        else:
            hi = mid
    return lo


def _fit_labels(ax, fig, items, min_pt, max_pt, value_weight):
    """Place a name/value block inside each cell, shrunk to fit; drop misfits.

    Text is measured at a probe size and scaled, since a text box's size is
    linear in its font size. Anything that cannot reach ``min_pt`` is dropped
    and reported back so the caller can list it under the chart -- a cell whose
    name did not fit still has to be identifiable somewhere.
    """
    renderer = fig.canvas.get_renderer()
    dropped = []

    for item in items:
        anchor, radius = item["anchor"], item["radius"]
        if radius <= 0:
            dropped.append(item)
            continue

        p0 = ax.transData.transform(anchor)
        p1 = ax.transData.transform(anchor + np.array([radius, radius]))
        r_px = float(min(abs(p1 - p0))) * 0.94  # breathing room off the edges

        for parts in ([item["name"], item["value"]], [item["value"]], []):
            parts = [p for p in parts if p]
            if not parts:
                dropped.append(item)
                break

            texts = []
            for part in parts:
                is_value = bool(item["value"]) and part == item["value"]
                texts.append(ax.text(
                    anchor[0], anchor[1], part, ha="center", va="center",
                    family=font_family(),
                    fontsize=_PROBE_PT * (1.0 if is_value else 0.82),
                    color=item["ink"], zorder=5,
                    fontweight=value_weight if is_value else "normal",
                ))

            sizes = [_text_size_px(t, renderer) for t in texts]
            total_w = max(w for w, _ in sizes)
            total_h = sum(h for _, h in sizes) + 0.30 * sizes[0][1] * (len(sizes) - 1)
            scale = _FIT_MARGIN * _fit_scale(item["poly_px"], p0, total_w, total_h, r_px)
            pt = float(np.clip(_PROBE_PT * scale, 0.0, max_pt))

            if pt < min_pt and len(parts) > 1:
                for t in texts:
                    t.remove()
                item["name_dropped"] = True
                continue  # retry with the value alone

            if pt < min_pt:
                for t in texts:
                    t.remove()
                dropped.append(item)
                break

            # lay the (now sized) lines out vertically around the anchor
            heights = [h * pt / _PROBE_PT for _, h in sizes]
            gap = 0.30 * heights[0]
            block = sum(heights) + gap * (len(heights) - 1)
            top = p0[1] + block / 2.0
            for t, (_, h0) in zip(texts, sizes):
                h = h0 * pt / _PROBE_PT
                t.set_fontsize(pt * (t.get_fontsize() / _PROBE_PT))
                y_px = top - h / 2.0
                t.set_position((anchor[0], ax.transData.inverted().transform((p0[0], y_px))[1]))
                top -= h + gap
            break

    return dropped


# --------------------------------------------------------------------------- #
# the chart
# --------------------------------------------------------------------------- #

def circle_area_chart(
    values,
    labels=None,
    groups=None,
    *,
    title=None,
    subtitle=None,
    note=None,
    theme="light",
    color_by=None,
    palette=None,
    group_order=None,
    show_values=True,
    show_labels=True,
    show_percent=False,
    value_format=None,
    prefix="",
    suffix="",
    figsize=(9.0, 9.0),
    dpi=200,
    seed=0,
    tol=0.004,
    max_iter=400,
    min_label_pt=5.5,
    max_label_pt=34.0,
    legend=True,
    list_unlabelled=True,
    gap=2.0,
    ax=None,
):
    """Render ``values`` as area-proportional polygons filling a circle.

    Parameters
    ----------
    values : sequence of positive numbers
    labels : sequence of str, optional
        Name drawn above each value.
    groups : sequence of str, optional
        Group per item; switches colour to categorical mode and adds a legend.
    title, subtitle, note : str, optional
        Headline, deck and footnote (source line).
    theme : {"light", "dark"}
    color_by : {"value", "group", "single"}, optional
        Defaults to ``"group"`` when ``groups`` is given, else ``"value"``.
    group_order : sequence of str, optional
        Which group takes which colour slot. Without it groups are ranked by
        total, so charting a subset can hand a group a different colour --
        pin the order when several charts have to agree.
    show_percent : bool
        Label with the share of the total instead of the raw value.
    value_format : callable, optional
        ``f(value) -> str``. Overrides ``prefix``/``suffix`` formatting.
    seed : int
        Layout seed - change it to get a different (equally valid) arrangement.
    tol : float
        Area accuracy, 0.004 = every cell within 0.4 % of its exact share.
    gap : float
        Width in points of the surface-coloured gap between cells.
    ax : matplotlib Axes, optional
        Draw into an existing axes instead of creating a figure.

    Returns
    -------
    (fig, ax)
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Polygon

    values = np.asarray(values, dtype=float).ravel()
    n = len(values)
    if labels is not None and len(labels) != n:
        raise ValueError("labels must have the same length as values")
    if groups is not None and len(groups) != n:
        raise ValueError("groups must have the same length as values")
    if theme not in THEMES:
        raise ValueError(f"theme must be one of {sorted(THEMES)}")
    tokens = THEMES[theme]

    if color_by is None:
        color_by = "group" if groups is not None else "value"
    if color_by == "group" and groups is None:
        raise ValueError('color_by="group" needs groups')

    # ---- layout -----------------------------------------------------------
    result = voronoi_treemap(values, radius=1.0, seed=seed, tol=tol, max_iter=max_iter)

    # ---- colours ----------------------------------------------------------
    order = np.argsort(np.argsort(-values, kind="stable"))  # 0 = largest
    if color_by == "single":
        base = palette[0] if palette else tokens["categorical"][0]
        colors = [base] * n
        group_colors = {}
    elif color_by == "value":
        ramp = palette or tokens["ramp"]
        # rank-spread so skewed data still separates; area carries the exact value
        t = order / max(n - 1, 1)
        colors = [_ramp_color(ramp, 1.0 - ti) for ti in t]
        group_colors = {}
    else:
        slots = palette or tokens["categorical"]
        totals = {}
        for g, v in zip(groups, values):
            totals[g] = totals.get(g, 0.0) + v
        ranked = sorted(totals, key=lambda g: -totals[g])
        if group_order:
            pinned = [g for g in group_order if g in totals]
            ranked = pinned + [g for g in ranked if g not in pinned]
        named, overflow = ranked[:len(slots)], ranked[len(slots):]
        slot_order = ranked
        group_colors = {g: slots[i] for i, g in enumerate(named)}
        for g in overflow:  # never cycle hues: the tail becomes "Other"
            group_colors[g] = tokens["other"]

        colors = []
        for g in set(groups):
            members = [i for i in range(n) if groups[i] == g]
            members.sort(key=lambda i: -values[i])
            for rank, i in enumerate(members):
                # tonal spread inside the group so neighbours never merge
                span = 0.34
                amount = (rank / max(len(members) - 1, 1)) * span - span / 3.0
                totals[(g, i)] = _tint(group_colors[g], amount)
        colors = [totals[(groups[i], i)] for i in range(n)]

    # ---- figure -----------------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        fig.patch.set_facecolor(tokens["surface"])
    else:
        fig = ax.figure
    ax.set_facecolor(tokens["surface"])
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_xlim(-1.06, 1.06)
    ax.set_ylim(-1.06, 1.06)

    for poly, color in zip(result.cells, colors):
        if len(poly) < 3:
            continue
        ax.add_patch(Polygon(poly, closed=True, facecolor=color,
                             edgecolor=tokens["surface"], linewidth=gap,
                             joinstyle="round", zorder=2))
    # hairline ring keeps the disc from floating on the surface
    ax.add_patch(Circle((0, 0), 1.0, facecolor="none",
                        edgecolor=tokens["border"], linewidth=1.2, zorder=4))

    # ---- headings ------------------------------------------------------
    top = 0.985
    if title:
        fig.text(0.055, top, title, ha="left", va="top", fontsize=21,
                 fontweight="bold", family=font_family(),
                 color=tokens["text_primary"])
        top -= 0.040
    if subtitle:
        fig.text(0.055, top, subtitle, ha="left", va="top", fontsize=12.5,
                 family=font_family(), color=tokens["text_secondary"])
        top -= 0.030
    # Reserve the footer strip before measuring labels: the axes box has to
    # be final before any text is fitted to it.
    has_legend = bool(legend) and color_by == "group"
    foot_lines = (1 if note else 0) + (2 if list_unlabelled else 0)
    note_y, legend_y = 0.018, 0.030
    foot_room = 0.030 + 0.024 * foot_lines
    legend_y += 0.024 * foot_lines
    if has_legend:
        foot_room += 0.050
    head_room = 1.0 - top if (title or subtitle) else 0.02
    if ax.get_subplotspec() is not None:
        fig.subplots_adjust(left=0.03, right=0.97,
                            top=1.0 - head_room - 0.015, bottom=foot_room)

    # ---- labels --------------------------------------------------------
    fmt = value_format or (lambda v: compact_number(v, prefix, suffix))
    total = float(values.sum())
    # equal aspect resizes the axes box during the draw, so lay it out
    # before anything is measured in pixels
    fig.canvas.draw()
    items = []
    for i, poly in enumerate(result.cells):
        if len(poly) < 3:
            continue
        anchor, radius = pole_of_inaccessibility(poly)
        if show_percent:
            text = f"{values[i] / total * 100:.1f}%"
        else:
            text = fmt(values[i])
        items.append({
            "index": i,
            "anchor": anchor,
            "radius": radius,
            "poly_px": ax.transData.transform(poly),
            "name": (labels[i] if labels is not None and show_labels else ""),
            "value": text if show_values else "",
            "ink": _ink_on(colors[i], tokens),
        })

    dropped = _fit_labels(ax, fig, items, min_label_pt, max_label_pt,
                          value_weight="bold")

    # ---- footnotes -----------------------------------------------------
    # Cells too small for a label still have to be readable somewhere.
    foot = []
    if note:
        foot.append(note)
    unnamed = [it for it in items if it.get("name_dropped") and it["name"]]
    if list_unlabelled:
        names = []
        for item in sorted(dropped + unnamed, key=lambda d: -values[d["index"]]):
            name = item["name"] or f"#{item['index'] + 1}"
            names.append(f"{name} {item['value']}".strip())
        if names:
            foot.append("Cells too small to name: " + _wrap(", ".join(names), 105))
    if foot:
        fig.text(0.055, note_y, "\n".join(foot), ha="left", va="bottom",
                 fontsize=8.5, family=font_family(),
                 color=tokens["text_muted"], linespacing=1.5)

    # ---- legend --------------------------------------------------------
    if has_legend:
        handles = []
        for g in sorted(set(groups), key=lambda g: slot_order.index(g)):
            handles.append(Line2D([], [], marker="o", linestyle="none",
                                  markersize=8, markerfacecolor=group_colors[g],
                                  markeredgecolor="none", label=str(g)))
        leg = fig.legend(handles=handles, loc="lower center",
                         ncol=min(len(handles), 6), frameon=False,
                         handletextpad=0.4, columnspacing=1.8,
                         bbox_to_anchor=(0.5, legend_y),
                         prop={"family": font_family(), "size": 10.5})
        for text in leg.get_texts():
            text.set_color(tokens["text_secondary"])

    ax._treemap_result = result  # handy for tests / further annotation
    return fig, ax


# --------------------------------------------------------------------------- #
# command line
# --------------------------------------------------------------------------- #

DEMO = [
    ("China", 1900, "Asia"), ("India", 573, "Asia"), ("U.S.", 451, "Americas"),
    ("Brazil", 204, "Americas"), ("Indonesia", 145, "Asia"), ("Turkiye", 92, "Middle East"),
    ("Russia", 87, "Europe"), ("Japan", 86, "Asia"), ("France", 82, "Europe"),
    ("Mexico", 77, "Americas"), ("Pakistan", 75, "Asia"), ("Spain", 64, "Europe"),
    ("Germany", 60, "Europe"), ("Canada", 58, "Americas"), ("Italy", 56, "Europe"),
    ("Vietnam", 56, "Asia"), ("Australia", 56, "Oceania"), ("Colombia", 53, "Americas"),
    ("Thailand", 48, "Asia"), ("S. Korea", 41, "Asia"), ("Egypt", 39, "Middle East"),
    ("UK", 35, "Europe"), ("Philippines", 35, "Asia"), ("Bangladesh", 34, "Asia"),
    ("Poland", 34, "Europe"), ("Iran", 31, "Middle East"), ("Algeria", 31, "Africa"),
    ("Malaysia", 29, "Asia"), ("Ukraine", 29, "Europe"), ("South Africa", 26, "Africa"),
    ("Peru", 22, "Americas"), ("Kenya", 22, "Africa"), ("Netherlands", 22, "Europe"),
    ("Saudi Arabia", 22, "Middle East"), ("Greece", 19, "Europe"),
    ("Cote d'Ivoire", 19, "Africa"), ("Uzbekistan", 18, "Asia"),
    ("Cambodia", 17, "Asia"), ("Morocco", 17, "Africa"), ("New Zealand", 17, "Oceania"),
]


def _read_csv(path):
    """Read ``label,value[,group]``; a header row is detected and skipped."""
    labels, values, groups = [], [], []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = [r for r in csv.reader(fh) if r and any(c.strip() for c in r)]
    if not rows:
        raise SystemExit(f"{path}: no data")
    try:
        float(rows[0][1] if len(rows[0]) > 1 else rows[0][0])
    except ValueError:
        rows = rows[1:]
    for row in rows:
        if len(row) == 1:
            labels.append("")
            values.append(float(row[0]))
        else:
            labels.append(row[0].strip())
            values.append(float(row[1]))
            if len(row) > 2 and row[2].strip():
                groups.append(row[2].strip())
    if groups and len(groups) != len(values):
        raise SystemExit(f"{path}: every row needs a group, or none should")
    return (labels if any(labels) else None), values, (groups or None)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Draw numbers as area-proportional polygons inside a circle.",
        epilog="examples:\n"
               "  python circle_area_chart.py 5 3 2 8 1 -o chart.png\n"
               "  python circle_area_chart.py --csv data.csv --title 'Revenue' -o out.png\n"
               "  python circle_area_chart.py --demo -o demo.png",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("numbers", nargs="*", type=float, help="the values to plot")
    p.add_argument("--csv", metavar="FILE", help="read label,value[,group] rows")
    p.add_argument("--labels", help="comma-separated names for the numbers")
    p.add_argument("--groups", help="comma-separated group per number")
    p.add_argument("--demo", action="store_true", help="plot the built-in example")
    p.add_argument("-o", "--output", default="circle_area_chart.png")
    p.add_argument("--title"), p.add_argument("--subtitle"), p.add_argument("--note")
    p.add_argument("--theme", choices=sorted(THEMES), default="light")
    p.add_argument("--color-by", choices=["value", "group", "single"])
    p.add_argument("--prefix", default="", help="e.g. $")
    p.add_argument("--suffix", default="", help="e.g. B or %%")
    p.add_argument("--percent", action="store_true", help="label shares, not values")
    p.add_argument("--no-values", action="store_true")
    p.add_argument("--size", type=float, default=9.0, help="figure side in inches")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tol", type=float, default=0.004)
    args = p.parse_args(argv)

    if args.demo:
        labels = [d[0] for d in DEMO]
        values = [d[1] * 1e9 for d in DEMO]  # the table is in billions of USD
        groups = [d[2] for d in DEMO]
        args.title = args.title or "The world's largest agricultural economies"
        args.subtitle = args.subtitle or \
            "Gross production value, 2024 - every polygon's area is proportional to its value"
        args.prefix = args.prefix or "$"
        args.note = args.note or "Illustrative figures, in USD."
    elif args.csv:
        labels, values, groups = _read_csv(args.csv)
    elif args.numbers:
        values = args.numbers
        labels = args.labels.split(",") if args.labels else None
        groups = args.groups.split(",") if args.groups else None
    else:
        p.error("give some numbers, --csv FILE, or --demo")

    matplotlib.use("Agg")
    fig, _ = circle_area_chart(
        values, labels, groups,
        title=args.title, subtitle=args.subtitle, note=args.note,
        theme=args.theme, color_by=args.color_by,
        prefix=args.prefix, suffix=args.suffix,
        show_percent=args.percent, show_values=not args.no_values,
        figsize=(args.size, args.size), dpi=args.dpi, seed=args.seed, tol=args.tol,
    )
    fig.savefig(args.output, facecolor=fig.get_facecolor())
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
