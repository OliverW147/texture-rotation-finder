import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import subprocess
import threading
import os
import sys
import multiprocessing
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CPU_EXE = os.path.join(SCRIPT_DIR, "height_match.exe")
GPU_EXE = os.path.join(SCRIPT_DIR, "height_match_gpu.exe")
SPEC_DIR = os.path.join(SCRIPT_DIR, "wgspec")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Surface Height Pattern Finder")
        self.resizable(True, True)
        self.matches_path = None
        self._tail_stop = None
        self._proc = None
        self._running = False
        self.start_time = None
        self._notes_visible = False
        self._build_ui()
        # Size to content, then lock minimum to that size
        self.update_idletasks()
        self.geometry(f"{self.winfo_reqwidth()}x{self.winfo_reqheight()}")
        self.minsize(self.winfo_reqwidth(), self.winfo_reqheight())

    def _build_ui(self):
        pad = dict(padx=8, pady=4)

        # --- Seed + centre ---
        top = ttk.LabelFrame(self, text="World / Search Centre")
        top.pack(fill="x", **pad)

        ttk.Label(top, text="World Seed:").grid(row=0, column=0, sticky="e", **pad)
        self.seed = ttk.Entry(top, width=28)
        self.seed.insert(0, "-377264746167088810")
        self.seed.grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(top, text="Centre X:").grid(row=1, column=0, sticky="e", **pad)
        self.cx = ttk.Entry(top, width=12)
        self.cx.insert(0, "0")
        self.cx.grid(row=1, column=1, sticky="w", **pad)

        ttk.Label(top, text="Centre Z:").grid(row=2, column=0, sticky="e", **pad)
        self.cz = ttk.Entry(top, width=12)
        self.cz.insert(0, "0")
        self.cz.grid(row=2, column=1, sticky="w", **pad)

        ttk.Label(top, text="Search Radius (blocks):").grid(row=3, column=0, sticky="e", **pad)
        self.radius = ttk.Entry(top, width=12)
        self.radius.insert(0, "300")
        self.radius.grid(row=3, column=1, sticky="w", **pad)

        ttk.Label(top, text="Orientation:").grid(row=4, column=0, sticky="e", **pad)
        self.orient = ttk.Combobox(top, width=22, state="readonly",
            values=["unknown (try all 8)", "0 - identity", "1 - rot 90",
                    "2 - rot 180", "3 - rot 270", "4 - reflect X",
                    "5 - reflect Z", "6 - transpose", "7 - anti-transpose"])
        self.orient.current(0)
        self.orient.grid(row=4, column=1, sticky="w", **pad)

        max_threads = multiprocessing.cpu_count()
        ttk.Label(top, text="Threads (CPU):").grid(row=5, column=0, sticky="e", **pad)
        self.threads = ttk.Spinbox(top, from_=1, to=max_threads, width=6)
        self.threads.set(max_threads)
        self.threads.grid(row=5, column=1, sticky="w", **pad)

        ttk.Label(top, text="Compute:").grid(row=6, column=0, sticky="e", **pad)
        self.compute = ttk.Combobox(top, width=18, state="readonly",
                                    values=["CPU (multi-thread)", "GPU (CUDA)"])
        self.compute.current(0)
        self.compute.grid(row=6, column=1, sticky="w", **pad)
        self.compute.bind("<<ComboboxSelected>>", self._on_compute_change)

        # --- Match options ---
        opts = ttk.LabelFrame(self, text="Match Options")
        opts.pack(fill="x", **pad)

        self.absolute = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts,
            text="Absolute heights (match exact Y; off = match relief shape only)",
            variable=self.absolute).pack(anchor="w", padx=6, pady=2)

        tolrow = ttk.Frame(opts)
        tolrow.pack(anchor="w", padx=6, pady=2)
        ttk.Label(tolrow, text="Tolerance (+- blocks per point):").pack(side="left")
        self.tol = ttk.Spinbox(tolrow, from_=0, to=16, width=5)
        self.tol.set(0)
        self.tol.pack(side="left", padx=6)

        maxrow = ttk.Frame(opts)
        maxrow.pack(anchor="w", padx=6, pady=2)
        self.stop_on_max = tk.BooleanVar(value=False)
        ttk.Checkbutton(maxrow, text="Stop after N matches:",
                        variable=self.stop_on_max).pack(side="left")
        self.maxn = ttk.Spinbox(maxrow, from_=1, to=100000, width=8)
        self.maxn.set(1)
        self.maxn.pack(side="left", padx=6)

        # --- Pattern offsets ---
        mid = ttk.LabelFrame(self, text="Surface Offsets:  dx,dy,dz")
        mid.pack(fill="x", **pad)

        hint_row = ttk.Frame(mid)
        hint_row.pack(fill="x", padx=6, pady=(2, 0))
        ttk.Label(hint_row, text="One per line.  dx,dz = XZ offset from centre;  dy = surface Y there.",
                  font=("Consolas", 8), foreground="#666666").pack(side="left")
        self._notes_btn = ttk.Button(hint_row, text="?", width=2, command=self._toggle_notes)
        self._notes_btn.pack(side="right")

        # Notes panel -- hidden by default, inserted here when shown
        self._notes_frame = ttk.Frame(mid, relief="groove", borderwidth=1)
        notes_text = (
            "Read the surface (grass-top) Y at a few offsets in-game,\n"
            "then this finds every world (x,z) whose terrain matches.\n\n"
            "Relief mode (default): matches shape only. Absolute Y can\n"
            "  be wrong; only up/down pattern is compared.\n"
            "Absolute mode: heights must match Y exactly (+- tol).\n\n"
            "Tips: use 6+ points in an asymmetric shape; avoid trees,\n"
            "  water, cave openings. Keep radius small (O(radius^2))."
        )
        ttk.Label(self._notes_frame, text=notes_text, font=("Consolas", 8),
                  foreground="#444444", justify="left").pack(anchor="w", padx=6, pady=4)

        self.offsets = tk.Text(mid, height=7, width=44, font=("Consolas", 10))
        self.offsets.insert("1.0",
            "-8,73,0\n-4,77,0\n0,75,0\n4,71,0\n8,69,0\n0,79,-8\n0,76,-4\n0,73,4\n0,71,8")
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
        self.output = scrolledtext.ScrolledText(out, height=6, font=("Consolas", 9),
                                                state="disabled", wrap="word")
        self.output.pack(fill="both", expand=True, padx=4, pady=4)
        self.output.tag_config("match", foreground="#00aa00", font=("Consolas", 9, "bold"))
        self.output.tag_config("err",   foreground="#cc0000")

        # --- Progress Bar & Status ---
        status_frame = ttk.Frame(self)
        status_frame.pack(fill="x", side="bottom", padx=8, pady=4)
        self.status_label = ttk.Label(status_frame, text="Ready")
        self.status_label.pack(side="left", padx=4)
        self.progress_bar = ttk.Progressbar(status_frame, orient="horizontal",
                                             length=180, mode="determinate")
        self.progress_bar.pack(side="right", padx=4, fill="x", expand=True)

    def _toggle_notes(self):
        if self._notes_visible:
            self._notes_frame.pack_forget()
            self._notes_btn.config(text="?")
        else:
            # insert between hint_row and the text widget
            self._notes_frame.pack(fill="x", padx=6, pady=(0, 2),
                                   before=self.offsets)
            self._notes_btn.config(text="x")
        self._notes_visible = not self._notes_visible

    def _parse_args(self):
        seed   = self.seed.get().strip()
        cx     = self.cx.get().strip()
        cz     = self.cz.get().strip()
        radius = self.radius.get().strip()
        orient_raw = self.orient.get().split(" - ")[0].split(" ")[0]
        orient = "-1" if orient_raw.startswith("unknown") else orient_raw
        threads = self.threads.get().strip()
        tol = self.tol.get().strip()

        for name, val in (("Seed", seed), ("Radius", radius)):
            if not val:
                raise ValueError(f"{name} is empty.")

        lines = self.offsets.get("1.0", "end").strip().splitlines()
        pts = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.count(",") != 2:
                raise ValueError(f"Bad offset line: '{line}'\nExpected: dx,dy,dz")
            dx, dy, dz = line.split(",")
            try:
                int(dx); int(dy); int(dz)
            except ValueError:
                raise ValueError(f"Non-integer in offset line: '{line}'")
            pts.append(f"{dx.strip()},{dz.strip()}:{dy.strip()}")

        if len(pts) < 2:
            raise ValueError("Need at least 2 offset points.")

        args = [seed, SPEC_DIR, cx, cz, radius, "--tol", tol, "--orient", orient]
        if not self.compute.get().startswith("GPU"):
            args += ["--threads", threads]
        if self.absolute.get():
            args.append("--absolute")
        if self.stop_on_max.get():
            args += ["--max", self.maxn.get().strip()]
        args += pts
        return args

    def _on_compute_change(self, event=None):
        use_gpu = self.compute.get().startswith("GPU")
        self.threads.config(state="disabled" if use_gpu else "normal")

    def _run(self):
        use_gpu = self.compute.get().startswith("GPU")
        exe = GPU_EXE if use_gpu else CPU_EXE
        if not os.path.exists(exe):
            name = "height_match_gpu.exe" if use_gpu else "height_match.exe"
            messagebox.showerror("Missing exe", f"{name} not found at:\n{exe}")
            return
        if not os.path.isdir(SPEC_DIR):
            messagebox.showerror("Missing spec", f"wgspec directory not found at:\n{SPEC_DIR}")
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
        self.start_time = time.time()
        self._proc = None

        self.matches_path = os.path.join(SCRIPT_DIR, "matches.txt")
        # Delete stale matches.txt before launch so the tailer never reads old results.
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
                # Start tailer only after the process exists so it waits for the
                # new matches.txt rather than potentially reading a stale one.
                threading.Thread(target=self._tail_matches, daemon=True).start()
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
        if self._tail_stop is not None:
            self.after(500, self._tail_stop.set)

    def _tail_matches(self):
        path = self.matches_path
        stop = self._tail_stop
        pos = 0
        deadline = time.time() + 10.0
        while not os.path.exists(path) and time.time() < deadline:
            if stop.is_set():
                return
            time.sleep(0.1)

        def emit_chunk(chunk):
            for line in chunk.splitlines(keepends=True):
                if line.startswith("MATCH "):
                    self.after(0, lambda l=line: self._log(l, tag="match"))

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
        if ("tiles " in line or "bands " in line) and "matches " in line:
            self._update_progress(line)
            return
        tag = "err" if ("Error" in line or "failed" in line) else None
        self.after(0, lambda l=line, t=tag: self._log(l, tag=t))

    def _update_progress(self, line):
        try:
            key = "tiles " if "tiles " in line else "bands "
            seg = line.split(key)[1].strip()
            cur, rest = seg.split("/", 1)
            tot = rest.split()[0]
            cur, tot = int(cur), int(tot)
            if self.start_time is None:
                self.start_time = time.time()
            elapsed = time.time() - self.start_time
            if cur > 0:
                eta = int((elapsed / cur) * (tot - cur))
                if eta >= 3600:
                    eta_str = f"{eta//3600:02d}:{(eta%3600)//60:02d}:{eta%60:02d}"
                else:
                    eta_str = f"{eta//60:02d}:{eta%60:02d}"
                label = "Tiles" if key == "tiles " else "Bands"
                txt = f"{label} {cur}/{tot} (ETA {eta_str})"
            else:
                label = "Tiles" if key == "tiles " else "Bands"
                txt = f"{label} {cur}/{tot}"
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
