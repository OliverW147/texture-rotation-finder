import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import subprocess
import threading
import os
import sys
import multiprocessing
import time  # Track time for ETA

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CPU_EXE = os.path.join(SCRIPT_DIR, "ore_match4.exe")
if not os.path.exists(CPU_EXE):
    CPU_EXE = os.path.join(SCRIPT_DIR, "ore_match.exe")

GPU_EXE = os.path.join(SCRIPT_DIR, "ore_match_gpu.exe")

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Ore Pattern Finder")
        self.geometry("520x720")
        self.resizable(True, True)
        self.matches_path = None
        self._tail_stop = None
        self._build_ui()
        self._running = False
        self.start_time = None

    def _build_ui(self):
        pad = dict(padx=8, pady=4)

        # --- Seed + centre ---
        top = ttk.LabelFrame(self, text="World / Centre Ore")
        top.pack(fill="x", **pad)

        ttk.Label(top, text="World Seed:").grid(row=0, column=0, sticky="e", **pad)
        self.seed = ttk.Entry(top, width=28)
        self.seed.insert(0, "-377264746167088810")
        self.seed.grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(top, text="Centre X:").grid(row=1, column=0, sticky="e", **pad)
        self.cx = ttk.Entry(top, width=10)
        self.cx.insert(0, "0")
        self.cx.grid(row=1, column=1, sticky="w", **pad)

        ttk.Label(top, text="Centre Y:").grid(row=2, column=0, sticky="e", **pad)
        self.cy = ttk.Entry(top, width=10)
        self.cy.insert(0, "0")
        self.cy.grid(row=2, column=1, sticky="w", **pad)

        ttk.Label(top, text="Centre Z:").grid(row=3, column=0, sticky="e", **pad)
        self.cz = ttk.Entry(top, width=10)
        self.cz.insert(0, "0")
        self.cz.grid(row=3, column=1, sticky="w", **pad)

        ttk.Label(top, text="Search Radius (blocks):").grid(row=4, column=0, sticky="e", **pad)
        self.radius = ttk.Entry(top, width=10)
        self.radius.insert(0, "1000")
        self.radius.grid(row=4, column=1, sticky="w", **pad)

        ttk.Label(top, text="Orientation:").grid(row=5, column=0, sticky="e", **pad)
        self.orient = ttk.Combobox(top, width=18, state="readonly",
            values=["unknown (try all 8)", "0 - (x,z)", "1 - (x,-z)", "2 - (-x,z)",
                    "3 - (-x,-z)", "4 - (z,x)", "5 - (z,-x)", "6 - (-z,x)", "7 - (-z,-x)"])
        self.orient.current(0)
        self.orient.grid(row=5, column=1, sticky="w", **pad)

        ttk.Label(top, text="Ore Type:").grid(row=6, column=0, sticky="e", **pad)
        self.ore_type = ttk.Combobox(top, width=16, state="readonly", values=[
            "iron", "coal", "copper", "gold", "redstone", "lapis", "diamond",
        ])
        self.ore_type.current(0)
        self.ore_type.grid(row=6, column=1, sticky="w", **pad)

        max_threads = multiprocessing.cpu_count()
        ttk.Label(top, text="Threads (CPU):").grid(row=7, column=0, sticky="e", **pad)
        self.threads = ttk.Spinbox(top, from_=1, to=max_threads, width=6)
        self.threads.set(max_threads)
        self.threads.grid(row=7, column=1, sticky="w", **pad)

        ttk.Label(top, text="Compute:").grid(row=8, column=0, sticky="e", **pad)
        self.compute = ttk.Combobox(top, width=16, state="readonly",
                                    values=["CPU (multi-thread)", "GPU (CUDA)"])
        self.compute.current(0)
        self.compute.grid(row=8, column=1, sticky="w", **pad)

        # Wire ore type change to update the Centre Y hint
        self.ore_type.bind("<<ComboboxSelected>>", self._on_ore_change)
        # Wire compute mode change to enable/disable thread spinner
        self.compute.bind("<<ComboboxSelected>>", self._on_compute_change)

        # --- Search Options ---
        opts = ttk.LabelFrame(self, text="Options")
        opts.pack(fill="x", **pad)

        self.stop_on_first = tk.BooleanVar(value=False)
        self.chk_stop = ttk.Checkbutton(opts, text="Stop on first exact match", variable=self.stop_on_first)
        self.chk_stop.pack(anchor="w", padx=6, pady=2)

        self.save_to_file = tk.BooleanVar(value=True)
        self.chk_save = ttk.Checkbutton(opts, text="Save matches to matches.txt", variable=self.save_to_file)
        self.chk_save.pack(anchor="w", padx=6, pady=2)

        # --- Speed & Compute notes ---
        # Explain why coal/copper are CPU-only and how to make a search fast.
        # Coal and copper seed huge, dense veins (coal: 40 veins of size 17 per
        # chunk; copper: 16 of size 20). On the GPU the match/generate kernel is
        # work-bound on that density, so the GPU engine refuses them and the CPU
        # engine is the supported path. Everything else runs well on the GPU.
        speed = ttk.LabelFrame(self, text="Speed & Compute Notes")
        speed.pack(fill="x", **pad)

        notes = (
            "Coal and copper are CPU-only.\n"
            "  Their veins are very dense and large (coal: 40 x size-17 per\n"
            "  chunk, copper: 16 x size-20), so the GPU kernel is work-bound\n"
            "  on them. The GPU engine rejects coal/copper -- run them on the\n"
            "  CPU engine (Compute = CPU) instead. All other ores run on GPU.\n"
            "\n"
            "Getting fast speed:\n"
            "  - Use Compute = GPU (CUDA) for iron/gold/redstone/lapis/diamond.\n"
            "    It is far faster than CPU for these ores.\n"
            "  - For coal/copper, use Compute = CPU and raise Threads to your\n"
            "    full core count (the default).\n"
            "  - Make the FIRST offset the rarest ore in your pattern (e.g.\n"
            "    diamond/lapis over iron). It anchors the search, so fewer\n"
            "    candidate blocks are scanned -- a big speedup.\n"
            "  - Keep the Search Radius only as large as you need; work grows\n"
            "    with the area (radius squared).\n"
            "  - A correct Centre Y and Orientation (if known) also cut work."
        )
        ttk.Label(speed, text=notes, font=("Consolas", 8),
                  foreground="#444444", justify="left").pack(anchor="w", padx=6, pady=4)

        # --- Pattern offsets ---
        mid = ttk.LabelFrame(self, text="Relative Ore Offsets")
        mid.pack(fill="both", expand=False, **pad)

        hint = ttk.Label(mid,
                         text="dx,dy,dz  or  dx,dy,dz,type\ntypes: iron  coal  copper  gold  redstone  lapis  diamond",
                         font=("Consolas", 8), foreground="#666666", justify="left")
        hint.pack(anchor="w", padx=6)

        self.offsets = tk.Text(mid, height=6, width=40, font=("Consolas", 10))
        self.offsets.insert("1.0", "1,0,0\n0,3,3\n1,4,3\n-1,4,3\n5,1,6\n9,4,4")
        self.offsets.pack(fill="both", expand=True, padx=6, pady=4)

        # --- Buttons ---
        btn_row = ttk.Frame(self)
        btn_row.pack(fill="x", **pad)
        self.run_btn = ttk.Button(btn_row, text="Run Search", command=self._run)
        self.run_btn.pack(side="left", padx=4)
        self.stop_btn = ttk.Button(btn_row, text="Stop", command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", padx=4)
        ttk.Button(btn_row, text="Clear Output", command=self._clear).pack(side="left", padx=4)

        # --- Output ---
        out = ttk.LabelFrame(self, text="Output")
        out.pack(fill="both", expand=True, **pad)
        self.output = scrolledtext.ScrolledText(out, height=10, font=("Consolas", 9),
                                                state="disabled", wrap="word")
        self.output.pack(fill="both", expand=True, padx=4, pady=4)

        self.output.tag_config("match", foreground="#00aa00", font=("Consolas", 9, "bold"))
        self.output.tag_config("near",  foreground="#cc7700")
        self.output.tag_config("err",   foreground="#cc0000")

        # --- Progress Bar & Status ---
        status_frame = ttk.Frame(self)
        status_frame.pack(fill="x", side="bottom", padx=8, pady=4)

        self.status_label = ttk.Label(status_frame, text="Ready")
        self.status_label.pack(side="left", padx=4)

        self.progress_bar = ttk.Progressbar(status_frame, orient="horizontal", length=180, mode="determinate")
        self.progress_bar.pack(side="right", padx=4, fill="x", expand=True)

    # Default Y centres per ore type (mid of typical range)
    _ORE_Y_DEFAULTS = {
        "iron":     "16",
        "coal":     "96",
        "copper":   "48",
        "gold":     "-32",
        "redstone": "-40",
        "lapis":    "0",
        "diamond":  "-34",
    }

    def _on_compute_change(self, event=None):
        use_gpu = self.compute.get().startswith("GPU")
        self.threads.config(state="disabled" if use_gpu else "normal")

    def _on_ore_change(self, event=None):
        ore = self.ore_type.get().split()[0]
        default_y = self._ORE_Y_DEFAULTS.get(ore, "0")
        self.cy.delete(0, "end")
        self.cy.insert(0, default_y)

    def _parse_args(self):
        seed   = self.seed.get().strip()
        cx     = self.cx.get().strip()
        cy     = self.cy.get().strip()
        cz     = self.cz.get().strip()
        radius = self.radius.get().strip()
        orient_raw = self.orient.get().split(" - ")[0].split(" ")[0]
        orient = "-1" if orient_raw.startswith("unknown") else orient_raw
        ore    = self.ore_type.get().strip().split()[0]

        use_gpu = self.compute.get().startswith("GPU")
        # The GPU engine supports every ore except coal/copper, whose veins are
        # very dense/large and leave the kernel work-bound (CPU is the path for
        # those). All other ores are wired up in the GPU engine's parseOreName.
        gpu_ores = {"iron","gold","redstone","lapis","diamond"}
        valid_ores = {"iron","coal","copper","gold","redstone","lapis","diamond"}

        if use_gpu and ore not in gpu_ores:
            raise ValueError(
                f"GPU mode does not support '{ore}'.\n"
                f"Coal and copper have very dense, large veins and are "
                f"work-bound on the GPU, so the GPU engine rejects them.\n"
                f"Switch Compute to CPU, or pick a GPU ore: "
                f"{', '.join(sorted(gpu_ores))}.")

        lines = self.offsets.get("1.0", "end").strip().splitlines()
        off_args = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3:
                off_args.append(",".join(parts))
            elif len(parts) == 4:
                if parts[3] not in valid_ores:
                    raise ValueError(f"Unknown ore type '{parts[3]}' in line: '{line}'\nValid: {', '.join(sorted(valid_ores))}")
                if use_gpu and parts[3] not in gpu_ores:
                    raise ValueError(
                        f"GPU mode does not support '{parts[3]}' (used in offset "
                        f"'{line}').\n"
                        f"Coal and copper have very dense veins and are work-bound "
                        f"on the GPU.\n"
                        f"Switch Compute to CPU, or use a GPU ore: "
                        f"{', '.join(sorted(gpu_ores))}.")
                off_args.append(",".join(parts))
            else:
                raise ValueError(f"Bad offset line: '{line}'\nExpected: dx,dy,dz  or  dx,dy,dz,oretype")

        if not off_args:
            raise ValueError("No offsets entered.")

        threads = self.threads.get().strip()

        if use_gpu:
            return [seed, cx, cy, cz, radius, orient, "--ore", ore] + off_args
        else:
            return [seed, cx, cy, cz, radius, orient, "--threads", threads, "--ore", ore] + off_args

    def _run(self):
        use_gpu = self.compute.get().startswith("GPU")
        exe = GPU_EXE if use_gpu else CPU_EXE

        if not os.path.exists(exe):
            name = "ore_match_gpu.exe" if use_gpu else "ore_match4.exe"
            messagebox.showerror("Missing exe", f"{name} not found at:\n{exe}")
            return

        try:
            extra_args = self._parse_args()
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
            return

        cmd = [exe] + extra_args
        self._log(f"Running: {os.path.basename(exe)} {' '.join(extra_args)}\n\n")

        self._running = True
        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.progress_bar.config(value=0, maximum=100)
        self.status_label.config(text="Starting search...")
        self.start_time = time.time()  # Record search start time
        self._proc = None

        # The binary owns matches.txt -- it truncates and writes MATCH lines as
        # they are found. The GUI tails that file to show the match list, so the
        # display is identical across the CPU and GPU engines and never drops a
        # line at high match volume. Remember the file's current size so the
        # tailer only reports MATCH lines this run appends.
        self.matches_path = os.path.join(SCRIPT_DIR, "matches.txt")
        self._tail_stop = threading.Event()
        if self.save_to_file.get():
            self._log("[Tailing matches.txt for exact matches]\n\n")
            threading.Thread(target=self._tail_matches, daemon=True).start()

        def worker():
            try:
                self._proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    cwd=SCRIPT_DIR, text=True, bufsize=1
                )
                while True:
                    line = self._proc.stdout.readline()
                    if not line:
                        break
                    self._log_line(line)
                self._proc.wait()
                code = self._proc.returncode
                self._log(f"\n[Process exited with code {code}]\n")
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
        # Let the tailer flush the final matches before it shuts down.
        if self._tail_stop is not None:
            self.after(500, self._tail_stop.set)

    def _tail_matches(self):
        # Follow matches.txt as the binary appends MATCH lines. The binary
        # truncates the file at startup, so we wait for it to (re)appear and
        # read from the beginning, skipping the header block.
        path = self.matches_path
        stop = self._tail_stop
        pos = 0
        first_seen = False
        # Wait for the binary to create/truncate the file (handles the brief
        # window before it opens matches.txt).
        deadline = time.time() + 10.0
        while not os.path.exists(path) and time.time() < deadline:
            if stop.is_set():
                return
            time.sleep(0.1)

        while True:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    f.seek(pos)
                    chunk = f.read()
                    pos = f.tell()
            except OSError:
                chunk = ""
            if chunk:
                for line in chunk.splitlines(keepends=True):
                    if "MATCH score=" not in line:
                        continue
                    self.after(0, lambda l=line: self._log(l, tag="match"))
                    if not first_seen:
                        first_seen = True
                        if self.stop_on_first.get() and self._running:
                            self.after(0, self._stop)
            if stop.is_set():
                # Final drain after stop, then exit.
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        f.seek(pos)
                        tail = f.read()
                    for line in tail.splitlines(keepends=True):
                        if "MATCH score=" in line:
                            self.after(0, lambda l=line: self._log(l, tag="match"))
                except OSError:
                    pass
                return
            time.sleep(0.25)

    def _log_line(self, line):
        # Handle progress line extraction (CPU: "Slab X/Y done", GPU: "Progress: X/Y rows done")
        if "Slab " in line and "done" in line:
            self._update_progress(line)
            if "best=" in line:
                line = line.split("best=")[0].rstrip() + "\n"

        elif "Progress: " in line and " rows done" in line:
            self._update_gpu_progress(line)
            if "best=" in line:
                line = line.split("best=")[0].rstrip() + "\n"

        # Clean "Best score" references from standard final report lines
        elif "Best score" in line or "best score" in line:
            if "All slabs done" in line:
                line = "All slabs done.\n"
            else:
                return

        # MATCH lines are surfaced by the matches.txt tailer, not from stdout,
        # so the display is engine-consistent and never drops at high volume.
        # Suppress the binary's stdout MATCH lines here to avoid showing them twice.
        if "MATCH score=" in line:
            return

        tag = None
        if "Error" in line or "Exception" in line:
            tag = "err"

        self.after(0, lambda l=line, t=tag: self._log(l, tag=t))

    def _update_gpu_progress(self, line):
        try:
            parts = line.split("Progress: ")[1].split(" rows done")[0]
            cur, tot = map(int, parts.split("/"))
            if self.start_time is None:
                self.start_time = time.time()
            elapsed = time.time() - self.start_time
            if cur > 0:
                eta_seconds = int((elapsed / cur) * (tot - cur))
                if eta_seconds >= 3600:
                    h, m, s = eta_seconds // 3600, (eta_seconds % 3600) // 60, eta_seconds % 60
                    eta_str = f"{h:02d}:{m:02d}:{s:02d}"
                else:
                    eta_str = f"{eta_seconds // 60:02d}:{eta_seconds % 60:02d}"
                status_text = f"GPU row {cur}/{tot} (ETA: {eta_str})"
            else:
                status_text = f"GPU row {cur}/{tot}"
            self.after(0, lambda: self.status_label.config(text=status_text))
            self.after(0, lambda: self.progress_bar.config(maximum=tot, value=cur))
        except Exception as e:
            print(f"[DEBUG] _update_gpu_progress: {e}", file=sys.stderr)

    def _update_progress(self, line):
        try:
            parts = line.split("Slab ")[1].split(" done")[0]
            cur, tot = map(int, parts.split("/"))
            
            if self.start_time is None:
                self.start_time = time.time()
                
            # ETA calculation
            elapsed = time.time() - self.start_time
            if cur > 0:
                avg_time_per_slab = elapsed / cur
                remaining_slabs = tot - cur
                eta_seconds = int(remaining_slabs * avg_time_per_slab)
                
                # Format ETA nicely based on duration
                if eta_seconds >= 3600:
                    h = eta_seconds // 3600
                    m = (eta_seconds % 3600) // 60
                    s = eta_seconds % 60
                    eta_str = f"{h:02d}:{m:02d}:{s:02d}"
                else:
                    m = eta_seconds // 60
                    s = eta_seconds % 60
                    eta_str = f"{m:02d}:{s:02d}"
                
                status_text = f"Slab {cur}/{tot} done (ETA: {eta_str})"
            else:
                status_text = f"Slab {cur}/{tot} done"
                
            self.after(0, lambda: self.status_label.config(text=status_text))
            self.after(0, lambda: self.progress_bar.config(maximum=tot, value=cur))
        except Exception as e:
            # Print debug errors to standard terminal if parsing fails
            print(f"[DEBUG] _update_progress exception: {e}", file=sys.stderr)

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