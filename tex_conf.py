"""
tex_conf.py -- read CoordsFinder .conf scan files and convert them into the
observation/range form used by tex_match and tex_match_gpu.

The .conf format, and the parsing and validation rules implemented here, come
from CoordsFinder by ALaggyDev:

    https://github.com/ALaggyDev/CoordsFinder

The reader below follows that project's src/config.rs closely, including its
accepted tokens, bounds checks and several of its error messages, so it is a
derived work under CoordsFinder's MIT licence, reproduced in full at the end
of this docstring. Thanks to that project for the format; this module exists
so observations already recorded in a .conf can be reused here without
retyping them.

A .conf holds a settings block plus a [filter] section of observations:

    algorithm = Vanilla-3
    scanOrder = spiral
    directions = [0]
    xRange = (-5000, 5000)
    yRange = (-60, 0)
    zRange = (-5000, 5000)
    errorTolerance = 0
    gpuTileSize = (16384, 16384)

    [filter]
    # x y z | variant [side]
    -6 0 0 | 3
    -6 0 -1 | 0 side

How each part is carried over:

  filter row  ->  observation "dx,dy,dz,rot,4"
                  A row marked `side` narrows four variants to two, which is
                  what modeff=2 expresses here, so it becomes
                  "dx,dy,dz,rot,4,2".  The two checks agree:
                  `variant & visible_mask == rotation` and
                  `(variant & (modeff-1)) == rot`.

  x/zRange    ->  centre + radius (see box_to_centre_radius)
  yRange      ->  --ymin / --ymax
  directions  ->  --facing indices (0, 90, 180, 270 -> 0, 1, 2, 3)

Settings with no equivalent here are skipped and listed in ConfScan.ignored:

  errorTolerance  this search is exact; see ConfScan.warnings
  cpu/gpu/cudaTileSize  tex_match_gpu sizes its own tiles from the search span
  scanOrder       tex_match_gpu always searches nearest-first from the centre
  verbose         these tools have their own output format

Only algorithm = Vanilla-3 is read.  Vanilla-1/2 and Sodium-1/2 select
different hash functions that tex_match does not implement, so those files are
declined rather than loaded with results that would not be meaningful.

----------------------------------------------------------------------------
The .conf parsing rules in this file are derived from CoordsFinder, used under
the following licence:

MIT License

Copyright (c) 2026 Laggy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
----------------------------------------------------------------------------
"""

import os
import re

# CoordsFinder algorithm name -> whether tex_match implements it.
SUPPORTED_ALGORITHM = "vanilla-3"

ALGORITHM_NOTES = {
    "vanilla-1": "Vanilla-1 (legacy 32-bit truncation + absolute modulo)",
    "vanilla-2": "Vanilla-2 (single-step LCG variant)",
    "sodium-1": "Sodium-1 (Sodium mod's mixer)",
    "sodium-2": "Sodium-2 (Sodium mod's mixer)",
}


class ConfError(Exception):
    """Raised for a .conf that cannot be represented as a tex_match search."""


class ConfScan:
    """A parsed .conf, reduced to what tex_match needs."""

    def __init__(self):
        self.path = None
        self.algorithm = "Vanilla-3"
        self.directions = [0]
        self.scan_order = None
        self.x_range = None      # (lo, hi) inclusive
        self.y_range = None
        self.z_range = None
        self.obs = []            # list of "dx,dy,dz,rot,mod[,modeff]"
        self.n_side = 0          # how many obs used the `side` marker
        self.ignored = []        # ["gpuTileSize = ...", ...] benign, FYI only
        self.warnings = []       # things that change the result, not just noise
        self.error_tolerance = 0

    # -- derived search box ------------------------------------------------

    def centre_radius(self):
        """(cx, cz, radius) covering both x and z ranges, minimising waste."""
        return box_to_centre_radius(self.x_range, self.z_range)

    def box_waste(self):
        """Warning text when the box is not square, else None.

        The search is a centred SQUARE, so a rectangular xRange/zRange has to
        be covered by the smallest square containing it. That square sticks
        out past the box, so matches outside the conf's stated range can be
        reported and extra area is scanned.
        """
        x_lo, x_hi = self.x_range
        z_lo, z_hi = self.z_range
        _, _, radius = self.centre_radius()
        side = 2 * radius + 1
        box = (x_hi - x_lo + 1) * (z_hi - z_lo + 1)
        square = side * side
        if square <= box:
            return None
        pct = 100.0 * (square - box) / box
        return (f"box is NOT SQUARE: covered by a {side}x{side} square, "
                f"{pct:.0f}% extra area scanned, and matches OUTSIDE the "
                "conf's x/z range can be reported")

    def facing_indices(self):
        """The conf's `directions` as tex_match facing indices (0-3)."""
        return sorted((d // 90) % 4 for d in self.directions)

    def facing_arg(self):
        """--facing value for the conf's `directions` list."""
        idx = self.facing_indices()
        if idx == [0, 1, 2, 3]:
            return "all4"
        return ",".join(str(i) for i in idx)


def box_to_centre_radius(x_range, z_range):
    """Smallest centred square covering both ranges.

    tex_match searches a square centred on (cx, cz) with a half-width of
    `radius`, so a rectangular .conf box has to be widened to a square.  The
    centre is placed at each range's own midpoint, which makes the required
    radius the LARGER of the two half-spans rather than anything bigger; any
    other centre would need a larger radius to still cover both ends.

    Midpoints are rounded so the covered span is never short: the returned
    square always contains the full box.
    """
    x_lo, x_hi = x_range
    z_lo, z_hi = z_range
    # floor-divide the midpoint, then grow the radius to cover the rounding,
    # so the box is covered exactly even when a span has odd length
    cx = (x_lo + x_hi) // 2
    cz = (z_lo + z_hi) // 2
    radius = max(x_hi - cx, cx - x_lo, z_hi - cz, cz - z_lo)
    return cx, cz, radius


# -- parsing ---------------------------------------------------------------

def _parse_pair(value, key):
    m = re.fullmatch(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", value.strip())
    if not m:
        raise ConfError(f"{key} must look like (lo, hi), got '{value}'")
    lo, hi = int(m.group(1)), int(m.group(2))
    if lo > hi:
        raise ConfError(f"{key} start must be <= end, got ({lo}, {hi})")
    return lo, hi


def _parse_directions(value):
    m = re.fullmatch(r"\[\s*(.*?)\s*\]", value.strip())
    if not m:
        raise ConfError(f"directions must look like [0, 90], got '{value}'")
    body = m.group(1).strip()
    if not body:
        raise ConfError("directions must not be empty")
    out = []
    for tok in body.split(","):
        tok = tok.strip()
        try:
            d = int(tok)
        except ValueError:
            raise ConfError(f"directions entry '{tok}' is not an integer")
        if d not in (0, 90, 180, 270):
            raise ConfError(f"directions entry {d} must be 0, 90, 180 or 270")
        if d in out:
            raise ConfError(f"duplicate direction {d}")
        out.append(d)
    return out


def _parse_filter_row(line, lineno):
    """'x y z | variant [side]' -> 'dx,dy,dz,rot,mod[,modeff]', is_side."""
    if "|" not in line:
        raise ConfError(f"line {lineno}: filter rows must be 'x y z | variant [side]'")
    coords, variant = line.split("|", 1)
    parts = coords.split()
    if len(parts) != 3:
        raise ConfError(f"line {lineno}: filter rows need exactly three coordinates")
    try:
        dx, dy, dz = (int(p) for p in parts)
    except ValueError:
        raise ConfError(f"line {lineno}: filter offsets must be integers")
    for name, v in (("x", dx), ("y", dy), ("z", dz)):
        if not -128 <= v <= 127:
            raise ConfError(f"line {lineno}: {name} offset {v} outside [-128, 127]")

    vparts = variant.split()
    if not vparts:
        raise ConfError(f"line {lineno}: filter row is missing a variant")
    try:
        rot = int(vparts[0])
    except ValueError:
        raise ConfError(f"line {lineno}: variant '{vparts[0]}' is not an integer")

    side = False
    if len(vparts) > 1:
        marker = vparts[1].lower()
        if marker in ("side", "true", "1"):
            side = True
        elif marker in ("normal", "false", "0"):
            side = False
        else:
            raise ConfError(f"line {lineno}: invalid side marker '{vparts[1]}'")
    if len(vparts) > 2:
        raise ConfError(f"line {lineno}: unexpected extra token '{vparts[2]}'")

    # side folds 4 variants to 2 states (their visible_mask=1) == modeff 2
    maximum = 1 if side else 3
    if not 0 <= rot <= maximum:
        raise ConfError(f"line {lineno}: variant {rot} exceeds maximum {maximum}")

    if side:
        return f"{dx},{dy},{dz},{rot},4,2", True
    return f"{dx},{dy},{dz},{rot},4", False


# settings this repo does not use; recorded so the GUI can say what it dropped
_IGNORED_KEYS = {
    "cputilesize": "tex_match sizes its own tiles",
    "gputilesize": "tex_match_gpu sizes its own tiles from the search span",
    "cudatilesize": "tex_match_gpu sizes its own tiles from the search span",
    "verbose": "this repo's tools have their own output format",
}


def parse(path):
    """Read a .conf and return a ConfScan.  Raises ConfError on anything
    that cannot be faithfully represented."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    scan = ConfScan()
    scan.path = path
    section = ""
    seen = set()
    algorithm_raw = None

    for index, original in enumerate(text.splitlines()):
        lineno = index + 1
        # CoordsFinder strips comments before parsing, so '#' is never data
        line = original.split("#", 1)[0].strip()
        if not line:
            continue

        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip().lower()
            continue

        if section == "filter":
            obs, is_side = _parse_filter_row(line, lineno)
            scan.obs.append(obs)
            scan.n_side += int(is_side)
            continue

        if "=" not in line:
            raise ConfError(f"line {lineno}: expected 'key = value', got '{line}'")
        key, value = line.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        if key in seen:
            raise ConfError(f"line {lineno}: duplicate setting '{key}'")
        seen.add(key)

        if key == "algorithm":
            algorithm_raw = value
        elif key == "directions":
            scan.directions = _parse_directions(value)
        elif key == "scanorder":
            scan.scan_order = value.strip().lower()
        elif key == "xrange":
            scan.x_range = _parse_pair(value, "xRange")
        elif key == "yrange":
            scan.y_range = _parse_pair(value, "yRange")
        elif key == "zrange":
            scan.z_range = _parse_pair(value, "zRange")
        elif key == "errortolerance":
            try:
                scan.error_tolerance = int(value)
            except ValueError:
                raise ConfError(f"line {lineno}: errorTolerance must be an integer")
            if scan.error_tolerance < 0:
                raise ConfError(f"line {lineno}: errorTolerance must be non-negative")
            # only a NON-ZERO tolerance changes the search: it lets a match
            # survive some wrong observations, which this exact sieve cannot do
            if scan.error_tolerance > 0:
                scan.warnings.append(
                    f"errorTolerance = {scan.error_tolerance} IGNORED: this search is "
                    f"exact, so up to {scan.error_tolerance} misread observation(s) "
                    "that the original would have tolerated will cause a MISS here")
            else:
                scan.ignored.append(f"{key} = {value}  (no tolerance requested)")
        elif key in _IGNORED_KEYS:
            scan.ignored.append(f"{key} = {value}  ({_IGNORED_KEYS[key]})")
        else:
            scan.ignored.append(f"{key} = {value}  (unrecognised)")

    # -- validation --------------------------------------------------------

    if algorithm_raw is None:
        raise ConfError("missing required setting 'algorithm'")
    algo = algorithm_raw.strip().lower().replace("_", "-")
    if algo != SUPPORTED_ALGORITHM:
        note = ALGORITHM_NOTES.get(algo, algorithm_raw)
        raise ConfError(
            f"this .conf selects {algorithm_raw}; tex_match reads Vanilla-3.\n"
            f"{note} is a different hash function, so the observations in\n"
            "this file would not line up with what tex_match computes."
        )
    scan.algorithm = algorithm_raw.strip()

    for name, val in (("xRange", scan.x_range), ("yRange", scan.y_range),
                      ("zRange", scan.z_range)):
        if val is None:
            raise ConfError(f"missing required setting '{name}'")
    if not scan.obs:
        raise ConfError("the [filter] section has no observations")

    # scanOrder: tex_match_gpu is always ring-ordered from the centre, which is
    # CoordsFinder's "spiral".  Only flag it when the file asked for something else.
    if scan.scan_order and scan.scan_order != "spiral":
        scan.ignored.append(
            f"scanOrder = {scan.scan_order}  (tex_match_gpu always searches "
            "nearest-first from the centre, i.e. spiral)")

    waste = scan.box_waste()
    if waste:
        scan.warnings.append(waste)

    return scan


def find_latest(directory):
    """Most recently modified .conf in `directory`, or None."""
    try:
        names = [n for n in os.listdir(directory) if n.lower().endswith(".conf")]
    except OSError:
        return None
    if not names:
        return None
    paths = [os.path.join(directory, n) for n in names]
    return max(paths, key=lambda p: os.path.getmtime(p))


# -- CLI (for testing) -----------------------------------------------------

def main():
    import sys
    if len(sys.argv) < 2:
        print("usage: python tex_conf.py <file.conf>")
        sys.exit(1)
    try:
        scan = parse(sys.argv[1])
    except ConfError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    cx, cz, radius = scan.centre_radius()
    print(f"# algorithm  {scan.algorithm}")
    print(f"# centre     ({cx}, {cz})  radius {radius}")
    print(f"# y          {scan.y_range[0]}..{scan.y_range[1]}")
    print(f"# facing     {scan.facing_arg()}  (directions {scan.directions})")
    print(f"# obs        {len(scan.obs)}  ({scan.n_side} side -> modeff=2)")
    for note in scan.warnings:
        print(f"# WARNING    {note}")
    for note in scan.ignored:
        print(f"# ignored    {note}")
    print(" ".join(scan.obs))


if __name__ == "__main__":
    main()
