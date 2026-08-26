import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import subprocess
import threading
import os
import sys
import multiprocessing
import math
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXE_CPU = os.path.join(SCRIPT_DIR, "tex_match.exe")
EXE_GPU = os.path.join(SCRIPT_DIR, "tex_match_gpu.exe")

# Blocks with multiple equal-weight blockstate variants in 26.1.2.
# Listed most-to-least useful for coordinate finding (common overworld blocks first).
# Each entry: (display_label, mod)
# Supported blocks with multiple equal-weight blockstate variants in 26.1.2.
# Listed most-to-least useful. mod = number of variants = nextInt argument.
BLOCK_LIST_TEXT = """\
Supported blocks (mod=4 unless noted):
  grass_block        dirt           sand
  netherrack  (mod=16: fully random orientation; one face
    read narrows it to 4 of 16 variants - enter those as
    a MASK observation, see below; 2 bits/obs like mod=4)
  red_sand           podzol         mycelium
  dirt_path          rooted_dirt    lily_pad
  sea_pickle         turtle_egg
  *_concrete_powder  (all 16 colours, mod=4)

Mirrored-variant blocks (use modeff=2 for side faces):
  stone   bedrock   sculk   deepslate
  infested_stone    infested_deepslate
  These have 4 variants but sometimes only 2 are visually
  distinct (normal vs mirrored). Use modeff=2 when you can
  only tell normal from mirrored, not the rotation.
  Deepslate: natural blocks only (axis=y); player-placed
  deepslate takes its axis from the placement direction.

Format per line:  dx,dy,dz,rot,mod[,modeff]
  mod=4  -> rot 0-3    mod=16 -> rot 0-15
  modeff (optional): effective mod for matching.
    modeff=2 on a mod=4 block means rot is 0 (normal)
    or 1 (mirrored); matches any variant where
    computed_rot % 2 == your rot.

Mask form:  dx,dy,dz,mHEX,mod
  Accepts any variant whose bit is set in the hex mask,
  e.g. 0,0,0,m807,16 = variants {0,1,2,11} of a mod-16
  block. Produced by the screenshot extractor
  (separate screenshot tool) for netherrack reads; works on CPU
  and GPU and with try-all-4-directions.\
"""

# ---------------------------------------------------------------------------
# How many observations give p50 / p95 / p99 probability of a unique result
# for a given search volume.
#
# Each top-face observation filters by 1/4, so after n obs the expected number
# of survivors in a volume V is V / 4^n.
# p50 unique  ->  V / 4^n < 1   ->  n > log2(V) / 2
# pXX unique  ->  P(survivors=0) = e^(-V/4^n) >= XX/100  (Poisson approx)
#              ->  V/4^n < -ln(1-XX/100)
# ---------------------------------------------------------------------------
def recommended_obs(radius, y_min=None, y_max=None, n_facings=1):
    """Return (p50, p95, p99) observation counts for the given search volume."""
    side = 2 * radius + 1
    if y_min is not None and y_max is not None:
        ny = max(1, y_max - y_min + 1)
    else:
        ny = side
    facings = max(1, n_facings)
    V = float(side) * float(side) * float(ny) * float(facings)
    if V <= 0:
        return 1, 1, 1

    def n_for_lambda(lam):
        # V / 4^n = lam  ->  n = log(V/lam) / log(4)
        return math.ceil(math.log(V / lam) / math.log(4))

    p50 = max(1, n_for_lambda(-math.log(0.50)))    # P(0 survivors) >= 0.50 -> lam = -ln(0.50)
    p95 = max(1, n_for_lambda(-math.log(0.95)))    # P(0 survivors) >= 0.95 -> lam = -ln(0.95)
    p99 = max(1, n_for_lambda(-math.log(0.99)))    # P(0 survivors) >= 0.99 -> lam = -ln(0.99)
    return p50, p95, p99


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Texture Rotation Coordinate Finder  (26.1.2+)")
        self.resizable(True, True)
        self.matches_path = None
        self._tail_stop = None
        self._proc = None
        self._running = False
        self.start_time = None
        self._notes_win = None
        self._build_ui()
        self.update_idletasks()
        req_w = self.winfo_reqwidth()
        req_h = self.winfo_reqheight()
        self.geometry(f"{req_w}x{req_h + 150}")
        self.minsize(req_w, req_h)
        self._update_rec_obs()

    def _build_ui(self):
        pad = dict(padx=8, pady=4)

        # Two-column body: inputs on the left, output on the right, so the
        # window stays wide-and-short instead of long-and-skinny. The status
        # bar spans the full width at the very bottom.
        body = ttk.Frame(self)
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body)
        left.pack(side="left", fill="y")
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)

        # --- Search centre ---
        top = ttk.LabelFrame(left, text="Search Centre")
        top.pack(fill="x", **pad)

        ttk.Label(top, text="Centre X:").grid(row=0, column=0, sticky="e", **pad)
        self.cx = ttk.Entry(top, width=14)
        self.cx.insert(0, "0")
        self.cx.grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(top, text="Centre Z:").grid(row=1, column=0, sticky="e", **pad)
        self.cz = ttk.Entry(top, width=14)
        self.cz.insert(0, "0")
        self.cz.grid(row=1, column=1, sticky="w", **pad)

        ttk.Label(top, text="Search Radius (blocks):").grid(row=2, column=0, sticky="e", **pad)
        self.radius = ttk.Entry(top, width=14)
        self.radius.insert(0, "5000")
        self.radius.grid(row=2, column=1, sticky="w", **pad)
        self.radius.bind("<FocusOut>",  lambda e: self._update_rec_obs())
        self.radius.bind("<Return>",    lambda e: self._update_rec_obs())
        self.radius.bind("<KeyRelease>", lambda e: self._update_rec_obs())

        ttk.Label(top, text="Y Min:").grid(row=3, column=0, sticky="e", **pad)
        self.ymin = ttk.Entry(top, width=14)
        self.ymin.insert(0, "62")
        self.ymin.grid(row=3, column=1, sticky="w", **pad)
        self.ymin.bind("<FocusOut>",   lambda e: self._update_rec_obs())
        self.ymin.bind("<KeyRelease>", lambda e: self._update_rec_obs())

        ttk.Label(top, text="Y Max:").grid(row=4, column=0, sticky="e", **pad)
        self.ymax = ttk.Entry(top, width=14)
        self.ymax.insert(0, "100")
        self.ymax.grid(row=4, column=1, sticky="w", **pad)
        self.ymax.bind("<FocusOut>",   lambda e: self._update_rec_obs())
        self.ymax.bind("<KeyRelease>", lambda e: self._update_rec_obs())

        ttk.Label(top, text="Direction:").grid(row=5, column=0, sticky="ne", **pad)
        facing_outer = ttk.Frame(top)
        facing_outer.grid(row=5, column=1, sticky="w", **pad)

        # One dropdown for every direction mode. Each entry maps to the
        # tex_match args it produces: (facing_arg, view_relative, n_facings).
        #
        # ANCHORED modes (no --view-relative): rot values are used exactly as
        # entered; --facing N just rotates the (dx,dz) offset pattern by N*90.
        # Use these when you know which way you were facing.
        #   facing 0 = offsets as entered, 1 = 90 deg CCW, 2 = 180, 3 = 270 CCW.
        #   "all 4 rotations" tries 0..3 at once (offsets only, rot anchored).
        #
        # VIEW-RELATIVE mode (--view-relative): direction unknown; tries all 4
        # camera directions, rotating offsets AND shifting rot in lockstep.
        # mod=4 blocks (and mod=16 mask obs) only; ~2x slower on GPU.
        #
        # FACING_MODES: ordered (key, label, facing_arg, view_relative, n_facings).
        # View-relative facing 0 is omitted: it equals anchored facing 0
        # (zero rot shift), so it would be a duplicate.
        self.FACING_MODES = [
            ("anchor0",   "As entered, facing known (anchored, facing 0)",      "0",    False, 1),
            ("anchor1",   "Anchored: rotate offsets 90 deg CCW (facing 1)",     "1",    False, 1),
            ("anchor2",   "Anchored: rotate offsets 180 deg (facing 2)",        "2",    False, 1),
            ("anchor3",   "Anchored: rotate offsets 270 deg CCW (facing 3)",    "3",    False, 1),
            ("anchorall", "Anchored: try all 4 offset rotations (rot fixed)",   "all4", False, 4),
            ("vr1",       "View-relative: camera turned 90 deg CW (facing 1)",  "1",    True,  1),
            ("vr2",       "View-relative: camera turned 180 deg (facing 2)",    "2",    True,  1),
            ("vr3",       "View-relative: camera turned 270 deg CW (facing 3)", "3",    True,  1),
            ("vrall",     "Try all 4 directions (view-relative, dir unknown)",  "all4", True,  4),
        ]
        self._facing_by_label = {lbl: (key, fa, vr, nf)
                                 for key, lbl, fa, vr, nf in self.FACING_MODES}

        self.facing_choice = tk.StringVar(value=self.FACING_MODES[0][1])
        self.facing_combo = ttk.Combobox(
            facing_outer, textvariable=self.facing_choice, state="readonly",
            width=46, values=[lbl for _, lbl, *_ in self.FACING_MODES])
        self.facing_combo.pack(anchor="w")
        self.facing_combo.bind("<<ComboboxSelected>>",
                               lambda e: self._update_rec_obs())

        ttk.Label(facing_outer,
            text="View-relative tries all 4 camera directions (~2x slower on\n"
                 "GPU); anchored modes use your rot values as entered.",
            font=("Consolas", 8), foreground="#666666",
            justify="left").pack(anchor="w", pady=(2, 0))

        max_threads = multiprocessing.cpu_count()
        ttk.Label(top, text="Threads (CPU only):").grid(row=6, column=0, sticky="e", **pad)
        self.threads = ttk.Spinbox(top, from_=1, to=max_threads, width=6)
        self.threads.set(max_threads)
        self.threads.grid(row=6, column=1, sticky="w", **pad)

        ttk.Label(top, text="Stop after N matches (GPU):").grid(row=7, column=0, sticky="e", **pad)
        stop_frame = ttk.Frame(top)
        stop_frame.grid(row=7, column=1, sticky="w", **pad)
        self.stop_n = ttk.Entry(stop_frame, width=8)
        self.stop_n.insert(0, "1")
        self.stop_n.pack(side="left")
        ttk.Label(stop_frame, text="(blank = search whole radius; GPU searches nearest-first)",
                  font=("Consolas", 8), foreground="#666666").pack(side="left", padx=(6, 0))

        ttk.Label(top, text="Mode:").grid(row=8, column=0, sticky="e", **pad)
        mode_frame = ttk.Frame(top)
        mode_frame.grid(row=8, column=1, sticky="w", **pad)
        self.mode_var = tk.StringVar(value="gpu" if os.path.exists(EXE_GPU) else "cpu")
        ttk.Radiobutton(mode_frame, text="CPU", variable=self.mode_var, value="cpu").pack(side="left")
        ttk.Radiobutton(mode_frame, text="GPU", variable=self.mode_var, value="gpu").pack(side="left", padx=(8, 0))
        if not os.path.exists(EXE_GPU):
            self.mode_var.set("cpu")

        # --- Observations ---
        mid = ttk.LabelFrame(left, text="Block Observations:  dx,dy,dz,rot,mod[,modeff]")
        mid.pack(fill="both", expand=True, **pad)

        hint_row = ttk.Frame(mid)
        hint_row.pack(fill="x", padx=6, pady=(2, 0))
        ttk.Label(hint_row,
            text="One per line.  mod=4 most blocks (rot 0-3),  mod=16 netherrack (rot 0-15).  No mod=2 in 26.1.2.",
            font=("Consolas", 8), foreground="#666666").pack(side="left")
        self._notes_btn = ttk.Button(hint_row, text="Help", width=5, command=self._toggle_notes)
        self._notes_btn.pack(side="right")

        # Help renders as wrapping paragraphs in a popup window (see
        # _toggle_notes). It is stored as ordered segments: "wrap" segments
        # reflow to the window width; "mono" segments (the column-aligned
        # block list / format spec) render verbatim in a fixed-width font.
        self._notes_segments = [
            ("wrap",
             "Texture rotations depend only on block (x,y,z) - not the world seed.\n"
             "Use them to find WHERE you are when coordinates are unknown.\n\n"
             "How to read rotations (26.1.2, vanilla):\n"
             "  Find the direction of your reference by using a non-random texture and looking at it from the same direction as in the reference\n"
             "  Face this direction in your own world and place the block until it appears the same as in your reference, then find this block's rot by testing all 4 rots using this program.\n"
             "  Now that you know which rotations to use, you can start entering information until the expected results number is at value that you like (usually <1)\n"
             "  If you can't find the direction using the reference, enable the try all 4 directions option (~2x slower), pick any direction, and then do the above steps. The x and z you enter still has to be correct relative to the rot you choose!\n"
             "  If you must use stone, deepslate, bedrock, or any other mirrored-variant block, but only have the side face (only narrows rot down to 0,2 or 1,3) you can use modeff=2 for that observation. If a lake or ocean is visible in the reference, you can infer the y coordinate of the anchor, saving lots of time."),
            ("mono", BLOCK_LIST_TEXT),
            ("wrap",
             "Tips:\n"
             "  GPU mode runs ~1000x faster than CPU for large radii.\n"
             "  Results are sorted nearest-to-centre first. The GPU also SEARCHES\n"
             "  nearest-first: set 'Stop after N matches' to quit as soon as the\n"
             "  N closest matches are confirmed (huge speedup if you trust your\n"
             "  centre guess).\n"
             "  Default Y range is just Centre Y (single level). Set Y Min/Max to search a range.\n"
             "  The search is seed-independent - works on any world.\n"
             "Direction modes (pick one in the Direction dropdown):\n"
             "  Anchored (facing 0-3): your rot values are used EXACTLY as entered;\n"
             "    the chosen facing only rotates the (dx,dz) offset pattern by\n"
             "    facing*90 (0 = as entered, 1 = 90 CCW, 2 = 180, 3 = 270 CCW).\n"
             "    Use when you know which way you were facing. Any mod allowed.\n"
             "  Anchored: try all 4 offset rotations: runs facings 0-3 at once with\n"
             "    rot fixed; useful if the offset orientation (not rot) is uncertain.\n"
             "  View-relative (facing 1-3): rot was read relative to the camera; the\n"
             "    search shifts rot in lockstep with the offset rotation (facing=f =\n"
             "    camera turned CW by f). mod=4 blocks / mod=16 mask obs only.\n"
             "  Try all 4 directions (view-relative): direction unknown. Tries all 4\n"
             "    camera directions the same way. ~2x slower on GPU.\n"
             "    (View-relative facing 0 is omitted: it equals anchored facing 0.)\n\n"
             "Screenshot extractor (separate tool, not bundled here):\n"
             "  Reads observations straight off a screenshot: click the crosshair and\n"
             "  the 4 corners of one block top face, scan, then paste the generated\n"
             "  string here with 'Try all 4 directions' enabled. Handles uneven\n"
             "  terrain (dy search), grass/dirt/sand/podzol/mycelium/dirt_path tops\n"
             "  and bottoms, stone-family tops and sides, and netherrack (mask obs).\n\n"
             "  The rot value is a painted digit from the pack - the digit rotates with\n"
             "  the BLOCK, not your camera, so its value reads the same from any view\n"
             "  direction (a rot-2 stone shows an upside-down '2' from the south, still\n"
             "  a 2). Only your dx/dz offsets are view-relative; the search rotates those\n"
             "  for you when 'Try all 4 directions' is on.\n\n"
             "Shared reference frame (important for 'Try all 4 directions'):\n"
             "  Pick ONE screen orientation (e.g. 'up-screen' = forward) and use it for\n"
             "  BOTH readings: dx,dz offsets measured with up-screen = -dz / right-screen\n"
             "  = +dx, and rot values judged against the base texture held in that same\n"
             "  orientation (what you call rot 0 defines the frame). Offsets and\n"
             "  rotations from different frames will never match."),
        ]

        self.offsets = tk.Text(mid, height=7, width=40, font=("Consolas", 10))
        self.offsets.bind("<KeyRelease>", lambda e: self._update_rec_obs())
        self.offsets.insert("1.0",
            "0,0,1,0,4\n1,0,0,0,4\n2,0,0,0,4\n3,0,0,0,4\n4,0,1,0,4\n1,0,4,0,4\n3,0,4,0,4")
        self.offsets.pack(fill="both", expand=True, padx=6, pady=4)

        # --- Recommended observations hint ---
        self._rec_label = ttk.Label(mid, text="", font=("Consolas", 8), foreground="#0055aa")
        self._rec_label.pack(anchor="w", padx=6, pady=(0, 4))

        # --- Buttons (under the left input column) ---
        btn_row = ttk.Frame(left)
        btn_row.pack(fill="x", **pad)
        self.run_btn = ttk.Button(btn_row, text="Run Search", command=self._run)
        self.run_btn.pack(side="left", padx=4)
        self.stop_btn = ttk.Button(btn_row, text="Stop", command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", padx=4)
        ttk.Button(btn_row, text="Clear Output", command=self._clear).pack(side="left", padx=4)

        # --- Output (right column, takes the vertical space) ---
        out = ttk.LabelFrame(right, text="Output")
        out.pack(fill="both", expand=True, **pad)
        self.output = scrolledtext.ScrolledText(out, width=60, height=8, font=("Consolas", 9),
                                                state="disabled", wrap="word")
        self.output.pack(fill="both", expand=True, padx=4, pady=4)
        self.output.tag_config("match", foreground="#00aa00", font=("Consolas", 9, "bold"))
        self.output.tag_config("err",   foreground="#cc0000")

        # --- Status ---
        status_frame = ttk.Frame(self)
        status_frame.pack(fill="x", side="bottom", padx=8, pady=4)
        self.status_label = ttk.Label(status_frame, text="Ready")
        self.status_label.pack(side="left", padx=4)
        self.progress_bar = ttk.Progressbar(status_frame, orient="horizontal",
                                             length=180, mode="determinate")
        self.progress_bar.pack(side="right", padx=4, fill="x", expand=True)

    def _facing_selection(self):
        # Resolve the current dropdown label to (key, facing_arg, view_rel, n_facings).
        return self._facing_by_label.get(
            self.facing_choice.get(), ("anchor0", "0", False, 1))

    def _update_rec_obs(self):
        try:
            r = int(self.radius.get().strip())
        except ValueError:
            self._rec_label.config(text="")
            return
        ymin_s = self.ymin.get().strip()
        ymax_s = self.ymax.get().strip()
        try:
            ym = int(ymin_s) if ymin_s else None
            yx = int(ymax_s) if ymax_s else None
        except ValueError:
            ym, yx = None, None
        _, _, _, n_facings = self._facing_selection()
        p50, p95, p99 = recommended_obs(r, ym, yx, n_facings=n_facings)

        # Compute expected match count from current observations
        side = 2 * r + 1
        ny = max(1, yx - ym + 1) if (ym is not None and yx is not None) else side
        V = float(side) * float(side) * float(ny) * float(max(1, n_facings))
        filter_denom = 1.0
        tokens = []
        for raw in self.offsets.get("1.0", "end").splitlines():
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            tokens.extend(raw.split())
        n_entered = 0
        for line in tokens:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            try:
                if len(parts) == 5 and parts[3].startswith("m"):
                    mask_val = int(parts[3][1:], 16)
                    mod_i = int(parts[4])
                    k = bin(mask_val).count("1")
                    if k > 0 and mod_i > 0:
                        filter_denom *= mod_i / k
                    n_entered += 1
                elif len(parts) in (5, 6):
                    modeff = int(parts[5]) if len(parts) == 6 else int(parts[4])
                    filter_denom *= max(1, modeff)
                    n_entered += 1
            except ValueError:
                continue
        expected = V / filter_denom if filter_denom > 0 else V

        self._rec_label.config(
            text=f"{n_entered} obs    Recommended obs: p99={p99}    Expected results: {expected:.4f}"
        )

    def _toggle_notes(self):
        # Help opens in its own scrollable popup window so it never reflows
        # or pushes down the main controls.
        win = getattr(self, "_notes_win", None)
        if win is not None and win.winfo_exists():
            win.destroy()
            self._notes_win = None
            return

        win = tk.Toplevel(self)
        win.title("Help - Texture Rotation Finder")
        win.transient(self)
        win.geometry("680x820")
        self._notes_win = win

        container = ttk.Frame(win)
        container.pack(fill="both", expand=True)
        canvas = tk.Canvas(container, highlightthickness=0)
        vbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vbar.set)
        vbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = ttk.Frame(canvas)
        inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        PAD = 8
        wrap_labels = []  # (label, is_mono) so we can keep mono full-width

        def add_paragraph(text, mono):
            font = ("Consolas", 8)
            lbl = ttk.Label(inner, text=text, font=font,
                            foreground="#444444", justify="left")
            lbl.pack(anchor="w", padx=PAD, pady=(0, 6))
            wrap_labels.append((lbl, mono))

        for kind, raw in self._notes_segments:
            if kind == "mono":
                # Column-aligned content: render verbatim, no reflow.
                add_paragraph(raw, True)
                continue
            # Reflow prose: split into paragraphs on blank lines, and treat a
            # trailing-colon header line as its own paragraph. Within a
            # paragraph, collapse newlines + indentation to single spaces so
            # the text wraps to the window width.
            for block in raw.split("\n\n"):
                lines = [ln.strip() for ln in block.split("\n")]
                buf = []
                for ln in lines:
                    if not ln:
                        continue
                    if ln.endswith(":") and buf:
                        add_paragraph(" ".join(buf), False)
                        buf = []
                    buf.append(ln)
                    if ln.endswith(":"):
                        add_paragraph(ln, False)
                        buf = []
                if buf:
                    add_paragraph(" ".join(buf), False)

        def _resize(e):
            # Match inner frame to canvas width and wrap prose to that width.
            canvas.itemconfigure(inner_id, width=e.width)
            wrap_w = max(200, e.width - 2 * PAD)
            for lbl, mono in wrap_labels:
                if not mono:
                    lbl.configure(wraplength=wrap_w)
        canvas.bind("<Configure>", _resize)
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        def _on_wheel(e):
            canvas.yview_scroll(-1 * (e.delta // 120), "units")
        canvas.bind_all("<MouseWheel>", _on_wheel)
        win.bind("<Destroy>",
                 lambda e: canvas.unbind_all("<MouseWheel>") if e.widget is win else None)

    def _parse_args(self):
        cx     = self.cx.get().strip()
        cz     = self.cz.get().strip()
        radius = self.radius.get().strip()
        threads = self.threads.get().strip()
        ymin   = self.ymin.get().strip()
        ymax   = self.ymax.get().strip()

        for name, val in (("Centre X", cx), ("Centre Z", cz), ("Radius", radius)):
            if not val:
                raise ValueError(f"{name} is empty.")
        try:
            int(cx); int(cz); int(radius); int(threads)
        except ValueError:
            raise ValueError("Centre X/Z, Radius, and Threads must be integers.")
        if ymin and not ymax:
            raise ValueError("Y Min is set but Y Max is empty. Set both or neither.")
        if ymax and not ymin:
            raise ValueError("Y Max is set but Y Min is empty. Set both or neither.")
        if ymin and ymax:
            try:
                ymin_i = int(ymin); ymax_i = int(ymax)
            except ValueError:
                raise ValueError("Y Min and Y Max must be integers.")
            if ymin_i > ymax_i:
                raise ValueError(f"Y Min ({ymin_i}) must be <= Y Max ({ymax_i}).")

        raw_lines = self.offsets.get("1.0", "end").strip().splitlines()
        # Support both one-per-line and space-separated observations on a single line.
        tokens = []
        for raw in raw_lines:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            tokens.extend(raw.split())
        lines = tokens
        obs_args = []
        has_mask_obs = False
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            # Mask observation: dx,dy,dz,mHEX,mod (produced by screenshot extractor)
            if len(parts) == 5 and parts[3].startswith("m"):
                try:
                    int(parts[0]); int(parts[1]); int(parts[2])
                    mask_val = int(parts[3][1:], 16)
                    mod_i = int(parts[4])
                except ValueError:
                    raise ValueError(f"Bad mask observation '{line}': expected dx,dy,dz,mHEX,mod")
                if mod_i not in (4, 16):
                    raise ValueError(f"mod must be 4 or 16 in mask obs '{line}', got {mod_i}")
                if mask_val == 0 or mask_val >> mod_i:
                    raise ValueError(f"mask 0x{mask_val:x} out of range for mod={mod_i} in '{line}'")
                obs_args.append(line)
                has_mask_obs = True
                continue
            if len(parts) not in (5, 6):
                raise ValueError(f"Bad observation '{line}': expected dx,dy,dz,rot,mod[,modeff]")
            try:
                dx_i, dy_i, dz_i, rot_i, mod_i = (int(p) for p in parts[:5])
                modeff_i = int(parts[5]) if len(parts) == 6 else mod_i
            except ValueError:
                raise ValueError(f"Non-integer value in '{line}'")
            if mod_i not in (4, 16):
                raise ValueError(f"mod must be 4 (most blocks) or 16 (netherrack), got {mod_i} in '{line}'")
            if modeff_i not in (1, 2, 4, 8, 16) or modeff_i > mod_i:
                raise ValueError(f"modeff must be a power of 2 and <= mod, got {modeff_i} in '{line}'")
            if not (0 <= rot_i < modeff_i):
                raise ValueError(f"rot must be 0-{modeff_i-1} when modeff={modeff_i}, got {rot_i} in '{line}'")
            if modeff_i == mod_i:
                obs_args.append(f"{dx_i},{dy_i},{dz_i},{rot_i},{mod_i}")
            else:
                obs_args.append(f"{dx_i},{dy_i},{dz_i},{rot_i},{mod_i},{modeff_i}")

        if len(obs_args) < 1:
            raise ValueError("Need at least 1 observation.")

        _, facing, view_rel, _ = self._facing_selection()
        if view_rel:
            # Every view-relative facing (single or all4) requires mod=4 obs
            # (or mod=16 mask obs), since rot shifts in lockstep with the view.
            for o in obs_args:
                p = o.split(",")
                # mod=4 normal obs and mod=16 mask obs are both allowed in view-relative mode
                if p[3].startswith("m"):
                    if p[4] not in ("4", "16"):
                        raise ValueError(f"Mask obs in '{o}' must be mod=4 or mod=16 for view-relative modes.")
                elif p[4] != "4":
                    raise ValueError("View-relative modes support only mod=4 "
                                     "blocks (and mod=16 mask observations); remove other mod=16 observations.")

        mode = self.mode_var.get()

        cy = "0"
        if mode == "gpu":
            args = [cx, cy, cz, radius]
        else:
            args = [cx, cy, cz, radius, threads]

        if ymin and ymax:
            args += ["--ymin", ymin, "--ymax", ymax]
        args += ["--facing", facing]
        if view_rel:
            args += ["--view-relative"]

        stop_s = self.stop_n.get().strip()
        if stop_s:
            try:
                stop_i = int(stop_s)
            except ValueError:
                raise ValueError("Stop after N matches must be an integer.")
            if stop_i < 1:
                raise ValueError("Stop after N matches must be >= 1.")
            if mode == "gpu":
                args += ["--stop", str(stop_i)]

        args += obs_args
        return args, mode

    def _run(self):
        cpu_ok = os.path.exists(EXE_CPU)
        gpu_ok = os.path.exists(EXE_GPU)
        mode = self.mode_var.get()

        if mode == "gpu" and not gpu_ok:
            messagebox.showerror("Missing exe",
                f"tex_match_gpu.exe not found.\nBuild with: build_tex.bat")
            return
        if mode == "cpu" and not cpu_ok:
            messagebox.showerror("Missing exe",
                f"tex_match.exe not found.\nBuild with: build_tex.bat")
            return

        try:
            extra_args, mode = self._parse_args()
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
            return

        exe = EXE_GPU if mode == "gpu" else EXE_CPU
        exe_name = "tex_match_gpu.exe" if mode == "gpu" else "tex_match.exe"
        cmd = [exe] + extra_args
        self._log(f"Running ({mode.upper()}): {exe_name} {' '.join(extra_args)}\n\n")

        self._running = True
        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.progress_bar.config(value=0, maximum=100)
        self.status_label.config(text="Starting search...")
        self.start_time = time.time()
        self._proc = None

        self.matches_path = os.path.join(SCRIPT_DIR, "matches.txt")
        try:
            os.remove(self.matches_path)
        except OSError:
            pass
        self._tail_stop = threading.Event()

        def worker():
            try:
                self._proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    cwd=SCRIPT_DIR, text=True, bufsize=1
                )
                threading.Thread(target=self._tail_matches, daemon=True).start()
                while True:
                    line = self._proc.stdout.readline()
                    if not line:
                        break
                    self._log_line(line)
                self._proc.wait()
                self._log(f"\n[Process exited with code {self._proc.returncode}]\n")
            except Exception as e:
                self._log(f"\n[Error: {e}]\n", tag="err")
            finally:
                self.after(0, self._done)

        threading.Thread(target=worker, daemon=True).start()

    def _stop(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            self._log("\n[Stopped by user]\n", tag="err")

    def _done(self):
        self._running = False
        self.run_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_label.config(text="Finished")
        if self._tail_stop is not None:
            self.after(500, self._tail_stop.set)

    def _tail_matches(self):
        path = self.matches_path
        stop = self._tail_stop
        pos = 0
        match_count = 0
        MATCH_DISPLAY_LIMIT = 100
        deadline = time.time() + 10.0
        while not os.path.exists(path) and time.time() < deadline:
            if stop.is_set():
                return
            time.sleep(0.1)

        def emit_chunk(chunk):
            nonlocal match_count
            for line in chunk.splitlines(keepends=True):
                if line.startswith("MATCH "):
                    if match_count < MATCH_DISPLAY_LIMIT:
                        self.after(0, lambda l=line: self._log(l, tag="match"))
                    elif match_count == MATCH_DISPLAY_LIMIT:
                        self.after(0, lambda: self._log(
                            f"[Display capped at {MATCH_DISPLAY_LIMIT} matches - see matches.txt for full results]\n",
                            tag="err"))
                    match_count += 1

        while True:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    f.seek(pos)
                    chunk = f.read()
                    pos = f.tell()
            except OSError:
                chunk = ""
            if chunk:
                emit_chunk(chunk)
            if stop.is_set():
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        f.seek(pos)
                        emit_chunk(f.read())
                except OSError:
                    pass
                return
            time.sleep(0.25)

    def _log_line(self, line):
        if "Progress: rows " in line:
            self._update_progress(line)
            return
        if line.startswith("MATCH "):
            return  # shown via tail
        tag = "err" if ("Error" in line or "error" in line) else None
        self.after(0, lambda l=line, t=tag: self._log(l, tag=t))

    def _update_progress(self, line):
        try:
            part = line.split("Progress: rows ")[1].strip()
            # Accept "X/Y" or "X/Y  (extra info)"
            nums = part.split()[0]
            cur, tot = map(int, nums.split("/"))
            elapsed = time.time() - (self.start_time or time.time())
            # Try to extract GPU ETA from line if present
            eta_str = ""
            if "ETA" in line:
                try:
                    eta_sec = int(line.split("ETA ")[1].split("s")[0])
                    if eta_sec >= 3600:
                        eta_str = f" ETA {eta_sec//3600:02d}:{(eta_sec%3600)//60:02d}:{eta_sec%60:02d}"
                    else:
                        eta_str = f" ETA {eta_sec//60:02d}:{eta_sec%60:02d}"
                except Exception:
                    pass
            elif cur > 0 and elapsed > 0:
                eta_sec = int((elapsed / cur) * (tot - cur))
                if eta_sec >= 3600:
                    eta_str = f" ETA {eta_sec//3600:02d}:{(eta_sec%3600)//60:02d}:{eta_sec%60:02d}"
                else:
                    eta_str = f" ETA {eta_sec//60:02d}:{eta_sec%60:02d}"
            txt = f"Rows {cur}/{tot}{eta_str}"
            self.after(0, lambda: self.status_label.config(text=txt))
            self.after(0, lambda: self.progress_bar.config(maximum=tot, value=cur))
        except Exception as e:
            print(f"[DEBUG] _update_progress: {e}", file=sys.stderr)

    def _log(self, text, tag=None):
        self.output.config(state="normal")
        if tag:
            self.output.insert("end", text, tag)
        else:
            self.output.insert("end", text)
        self.output.see("end")
        self.output.config(state="disabled")

    def _clear(self):
        self.output.config(state="normal")
        self.output.delete("1.0", "end")
        self.output.config(state="disabled")


if __name__ == "__main__":
    app = App()
    app.mainloop()
