"""Tkinter GUI for Automated Accuracy Test Bench Software."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

from PIL import Image, ImageTk

from .analysis import (
    analyze,
    archive_log_to_run,
    confusion_matrix_labels,
    find_sd_csv,
    load_inference_csv,
    load_saved_run_analysis,
    write_report_csv,
)
from .dataset import discover_classes, sample_test_set
from .metadata import (
    RunMetadata,
    chart_path,
    create_and_save_run,
    default_data_dir,
    load_run,
    resolve_run_dir,
    results_path,
    save_run,
)

# Soft lab-bench look (not purple / cream / broadsheet defaults).
BG = "#e8eef2"
PANEL = "#f7fafc"
ACCENT = "#1f6f5b"
ACCENT_DARK = "#155445"
TEXT = "#1a2b33"
MUTED = "#5a6f7a"


class AATBSApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("AATBS — Automated Accuracy Test Bench Software")
        self.geometry("1280x860")
        self.minsize(1100, 720)
        self.configure(bg=BG)

        self.run: Optional[RunMetadata] = None
        self.run_dir: Optional[Path] = None
        self.run_path: Optional[Path] = None  # metadata.json inside run_dir
        self.current_pos = 0  # 0-based into run.images
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._chart_photo: Optional[ImageTk.PhotoImage] = None
        self._last_report_run_dir: Optional[Path] = None
        self._chart_source_path: Optional[Path] = None
        self._chart_resize_job: Optional[str] = None

        self._style()
        self._build()

    def _style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("Card.TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("Title.TLabel", background=BG, foreground=ACCENT_DARK, font=("Segoe UI", 18, "bold"))
        style.configure("Sub.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 10))
        style.configure("Status.TLabel", background=PANEL, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("TButton", font=("Segoe UI", 10), padding=6)
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=8)
        style.map(
            "Accent.TButton",
            background=[("!disabled", ACCENT), ("pressed", ACCENT_DARK)],
            foreground=[("!disabled", "white")],
        )
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(14, 8), font=("Segoe UI", 10))
        style.configure("TLabelframe", background=BG, foreground=TEXT)
        style.configure("TLabelframe.Label", background=BG, foreground=ACCENT_DARK, font=("Segoe UI", 10, "bold"))
        style.configure("Treeview", font=("Segoe UI", 9), rowheight=24)
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))

    def _build(self) -> None:
        header = ttk.Frame(self, padding=(16, 12, 16, 4))
        header.pack(fill="x")
        ttk.Label(header, text="AATBS", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Class-folder dataset → on-screen capture → SD inference import → accuracy analysis",
            style="Sub.TLabel",
        ).pack(anchor="w")

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=12, pady=(4, 12))

        self.setup_tab = ttk.Frame(self.notebook, padding=12)
        self.capture_tab = ttk.Frame(self.notebook, padding=12)
        self.analysis_tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(self.setup_tab, text="1. Setup")
        self.notebook.add(self.capture_tab, text="2. Capture Display")
        self.notebook.add(self.analysis_tab, text="3. Analysis")

        self._build_setup()
        self._build_capture()
        self._build_analysis()

    # ------------------------------------------------------------------ Setup
    def _build_setup(self) -> None:
        left = ttk.Frame(self.setup_tab)
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))
        right = ttk.Frame(self.setup_tab, style="Card.TFrame")
        right.pack(side="right", fill="both", expand=True, padx=(8, 0))

        form = ttk.LabelFrame(left, text="Test set configuration", padding=12)
        form.pack(fill="x")

        self.dataset_var = tk.StringVar()
        self.ipc_var = tk.StringVar(value="5")
        self.seed_var = tk.StringVar(value="42")

        ttk.Label(form, text="Dataset folder").grid(row=0, column=0, sticky="w", pady=4)
        ds_row = ttk.Frame(form)
        ds_row.grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Entry(ds_row, textvariable=self.dataset_var, width=48).pack(side="left", fill="x", expand=True)
        ttk.Button(ds_row, text="Browse…", command=self._browse_dataset).pack(side="left", padx=(6, 0))

        ttk.Label(form, text="Images per class").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(form, textvariable=self.ipc_var, width=10).grid(row=1, column=1, sticky="w", pady=4)

        ttk.Label(form, text="Random seed (optional)").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(form, textvariable=self.seed_var, width=10).grid(row=2, column=1, sticky="w", pady=4)

        form.columnconfigure(1, weight=1)

        btns = ttk.Frame(left)
        btns.pack(fill="x", pady=12)
        ttk.Button(btns, text="Scan dataset", command=self._scan_dataset).pack(side="left")
        ttk.Button(btns, text="Generate test set", style="Accent.TButton", command=self._generate_test_set).pack(
            side="left", padx=8
        )
        ttk.Button(btns, text="Load existing run folder…", command=self._load_run_dialog).pack(side="left")

        ttk.Label(left, text="Detected classes (from folder names)", style="Sub.TLabel").pack(anchor="w", pady=(4, 2))
        self.class_list = tk.Listbox(left, height=14, font=("Segoe UI", 10), bg="white", fg=TEXT)
        self.class_list.pack(fill="both", expand=True)

        ttk.Label(right, text="Run metadata", style="Card.TLabel", font=("Segoe UI", 11, "bold")).pack(
            anchor="w", padx=12, pady=(12, 4)
        )
        self.meta_text = tk.Text(
            right,
            wrap="word",
            font=("Consolas", 10),
            bg="white",
            fg=TEXT,
            relief="flat",
            padx=10,
            pady=10,
        )
        self.meta_text.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.meta_text.insert("1.0", "No run loaded.\n\nGenerate a test set or load a previous run.")
        self.meta_text.configure(state="disabled")

    def _browse_dataset(self) -> None:
        path = filedialog.askdirectory(title="Select dataset folder")
        if path:
            self.dataset_var.set(path)
            self._scan_dataset(announce=False)

    def _scan_dataset(self, announce: bool = True) -> None:
        root = self.dataset_var.get().strip()
        if not root:
            if announce:
                messagebox.showwarning("Dataset", "Choose a dataset folder first.")
            return
        try:
            classes = discover_classes(Path(root))
        except (FileNotFoundError, ValueError) as exc:
            messagebox.showerror("Dataset", str(exc))
            return

        self.class_list.delete(0, "end")
        for name, paths in sorted(classes.items()):
            self.class_list.insert("end", f"{name}  ({len(paths)} images)")
        if announce:
            messagebox.showinfo("Dataset", f"Found {len(classes)} classes.")

    def _generate_test_set(self) -> None:
        root = self.dataset_var.get().strip()
        if not root:
            messagebox.showwarning("Dataset", "Choose a dataset folder first.")
            return
        try:
            ipc = int(self.ipc_var.get().strip())
        except ValueError:
            messagebox.showerror("Input", "Images per class must be an integer.")
            return
        seed_raw = self.seed_var.get().strip()
        seed = int(seed_raw) if seed_raw else None

        try:
            images = sample_test_set(
                Path(root),
                images_per_class=ipc,
                seed=seed,
            )
            meta, run_dir, path = create_and_save_run(
                Path(root), images, images_per_class=ipc, seed=seed
            )
        except (FileNotFoundError, ValueError, OSError) as exc:
            messagebox.showerror("Generate", str(exc))
            return

        self._set_run(meta, run_dir, path)
        self.notebook.select(self.capture_tab)
        messagebox.showinfo(
            "Test set ready",
            f"Generated {meta.total_images} images across {len(meta.classes)} classes.\n"
            f"Run folder: {run_dir.name} (run #{meta.run_number})",
        )

    def _load_run_dialog(self) -> None:
        initial = default_data_dir()
        initial.mkdir(parents=True, exist_ok=True)
        folder = filedialog.askdirectory(
            title="Select run folder (e.g. data/run1_20260812)",
            initialdir=str(initial),
        )
        if not folder:
            return
        try:
            run_dir = resolve_run_dir(Path(folder))
            meta = load_run(run_dir)
        except (OSError, ValueError, KeyError, FileNotFoundError) as exc:
            messagebox.showerror("Load", str(exc))
            return
        self._set_run(meta, run_dir, run_dir / "metadata.json")
        messagebox.showinfo(
            "Loaded",
            f"Loaded {run_dir.name} ({meta.total_images} images).",
        )

    def _set_run(self, meta: RunMetadata, run_dir: Path, path: Path) -> None:
        self.run = meta
        self.run_dir = Path(run_dir)
        self.run_path = Path(path)
        self.current_pos = 0
        self._refresh_meta_panel()
        self._refresh_capture_view()
        self.analysis_run_var.set(str(self.run_dir))

    def _refresh_meta_panel(self) -> None:
        self.meta_text.configure(state="normal")
        self.meta_text.delete("1.0", "end")
        if not self.run:
            self.meta_text.insert("1.0", "No run loaded.")
            self.meta_text.configure(state="disabled")
            return
        m = self.run
        lines = [
            f"Run folder:       {m.folder_name or (self.run_dir.name if self.run_dir else '?')}",
            f"Run number:       {m.run_number}",
            f"Run ID:           {m.run_id}",
            f"Created:          {m.created_at}",
            f"Dataset:          {m.dataset_root}",
            f"Images / class:   {m.images_per_class}",
            f"Seed:             {m.seed}",
            f"Total images:     {m.total_images}",
            f"Captured so far:  {m.captured_count()} / {m.total_images}",
            f"Classes ({len(m.classes)}):",
        ]
        for c in m.classes:
            count = sum(1 for img in m.images if img.class_name == c)
            lines.append(f"  - {c}: {count}")
        if self.run_dir:
            lines.append(f"\nRun folder path:\n{self.run_dir}")
        self.meta_text.insert("1.0", "\n".join(lines))
        self.meta_text.configure(state="disabled")

    # ---------------------------------------------------------------- Capture
    def _build_capture(self) -> None:
        top = ttk.Frame(self.capture_tab)
        top.pack(fill="x")

        self.capture_status = ttk.Label(top, text="No test set loaded.", style="Sub.TLabel")
        self.capture_status.pack(anchor="w")

        self.image_frame = tk.Frame(self.capture_tab, bg="#0f1a1f", highlightthickness=0)
        self.image_frame.pack(fill="both", expand=True, pady=8)
        self.image_label = tk.Label(self.image_frame, bg="#0f1a1f", fg="white", text="Image will appear here")
        self.image_label.pack(fill="both", expand=True)

        controls = ttk.Frame(self.capture_tab)
        controls.pack(fill="x")

        ttk.Button(controls, text="◀ Previous", command=self._prev_image).pack(side="left")
        ttk.Button(controls, text="Mark captured + Next ▶", style="Accent.TButton", command=self._mark_and_next).pack(
            side="left", padx=8
        )
        ttk.Button(controls, text="Mark as pending", command=self._mark_pending).pack(side="left")
        ttk.Button(controls, text="Next ▶", command=self._next_image).pack(side="left", padx=8)
        ttk.Button(controls, text="Jump to first uncaptured", command=self._jump_uncaptured).pack(side="left")

    def _refresh_capture_view(self) -> None:
        if not self.run or not self.run.images:
            self.capture_status.configure(text="No test set loaded.")
            self.image_label.configure(image="", text="Image will appear here")
            self._photo = None
            return

        pos = max(0, min(self.current_pos, len(self.run.images) - 1))
        self.current_pos = pos
        img = self.run.images[pos]
        flag = "CAPTURED" if img.captured else "PENDING"
        self.capture_status.configure(
            text=(
                f"Image {img.index} / {self.run.total_images}   |   "
                f"class: {img.class_name}   |   {img.filename}   |   {flag}   |   "
                f"progress {self.run.captured_count()}/{self.run.total_images}"
            )
        )
        self._show_image(Path(img.path))
        self._refresh_meta_panel()

    def _show_image(self, path: Path) -> None:
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                # Fit into current panel while keeping aspect ratio.
                self.image_frame.update_idletasks()
                max_w = max(self.image_frame.winfo_width() - 20, 400)
                max_h = max(self.image_frame.winfo_height() - 20, 300)
                im.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
                self._photo = ImageTk.PhotoImage(im)
                self.image_label.configure(image=self._photo, text="")
        except OSError as exc:
            self.image_label.configure(
                image="",
                text=f"Could not open image (corrupt or invalid file):\n{path}\n{exc}",
            )
            self._photo = None

    def _persist_run(self) -> None:
        if self.run and self.run_dir:
            self.run_path = save_run(self.run, self.run_dir)

    def _prev_image(self) -> None:
        if not self.run:
            return
        self.current_pos = max(0, self.current_pos - 1)
        self._refresh_capture_view()

    def _next_image(self) -> None:
        if not self.run:
            return
        self.current_pos = min(len(self.run.images) - 1, self.current_pos + 1)
        self._refresh_capture_view()

    def _mark_and_next(self) -> None:
        if not self.run:
            return
        img = self.run.images[self.current_pos]
        self.run.mark_captured(img.index, True)
        self._persist_run()
        if self.current_pos < len(self.run.images) - 1:
            self.current_pos += 1
        self._refresh_capture_view()

    def _mark_pending(self) -> None:
        if not self.run:
            return
        img = self.run.images[self.current_pos]
        self.run.mark_captured(img.index, False)
        self._persist_run()
        self._refresh_capture_view()

    def _jump_uncaptured(self) -> None:
        if not self.run:
            return
        for i, img in enumerate(self.run.images):
            if not img.captured:
                self.current_pos = i
                self._refresh_capture_view()
                return
        messagebox.showinfo("Capture", "All images are marked as captured.")

    # --------------------------------------------------------------- Analysis
    def _build_analysis(self) -> None:
        form = ttk.LabelFrame(self.analysis_tab, text="Import SD card results", padding=10)
        form.pack(fill="x")

        self.analysis_run_var = tk.StringVar()
        self.sd_var = tk.StringVar()

        ttk.Label(form, text="Run folder").grid(row=0, column=0, sticky="w", pady=2)
        run_row = ttk.Frame(form)
        run_row.grid(row=0, column=1, sticky="ew", pady=2)
        ttk.Entry(run_row, textvariable=self.analysis_run_var).pack(side="left", fill="x", expand=True)
        ttk.Button(run_row, text="Browse…", command=self._browse_analysis_run).pack(side="left", padx=(6, 0))

        ttk.Label(form, text="SD card folder or .csv").grid(row=1, column=0, sticky="w", pady=2)
        sd_row = ttk.Frame(form)
        sd_row.grid(row=1, column=1, sticky="ew", pady=2)
        ttk.Entry(sd_row, textvariable=self.sd_var).pack(side="left", fill="x", expand=True)
        ttk.Button(sd_row, text="Browse…", command=self._browse_sd).pack(side="left", padx=(6, 0))
        form.columnconfigure(1, weight=1)

        action_row = ttk.Frame(form)
        action_row.grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Button(action_row, text="Run analysis", style="Accent.TButton", command=self._run_analysis).pack(
            side="left"
        )
        ttk.Button(
            action_row,
            text="Load old analysis from run folder…",
            command=self._load_old_analysis,
        ).pack(side="left", padx=8)

        # Charts take most of the vertical space.
        chart_wrap = ttk.Frame(self.analysis_tab)
        chart_wrap.pack(fill="both", expand=True, pady=(6, 4))
        ttk.Label(chart_wrap, text="Accuracy charts", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.chart_label = tk.Label(chart_wrap, bg="#ffffff", text="Charts appear after analysis", anchor="center")
        self.chart_label.pack(fill="both", expand=True)
        self.chart_label.bind("<Configure>", self._on_chart_resize)

        bottom = ttk.Frame(self.analysis_tab)
        bottom.pack(fill="x", pady=(0, 4))

        left = ttk.Frame(bottom)
        left.pack(side="left", fill="both", expand=True, padx=(0, 6))
        right = ttk.Frame(bottom)
        right.pack(side="right", fill="both", expand=True)

        ttk.Label(left, text="Summary", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.summary_text = tk.Text(left, wrap="word", font=("Consolas", 9), height=8, bg="white", fg=TEXT)
        self.summary_text.pack(fill="both", expand=True)
        self.summary_text.insert(
            "1.0",
            "Import the SD card .csv after the capture session,\n"
            "or load an old analysis by browsing a data/runN_YYYYMMDD folder.",
        )
        self.summary_text.configure(state="disabled")

        ttk.Label(right, text="Per-image results", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        tree_frame = ttk.Frame(right)
        tree_frame.pack(fill="both", expand=True)
        cols = ("index", "truth", "pred", "gate", "ok", "ms")
        self.result_tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=8)
        headings = {
            "index": "ID",
            "truth": "Ground truth",
            "pred": "Disease pred",
            "gate": "Gate",
            "ok": "Correct",
            "ms": "Infer ms",
        }
        widths = {"index": 50, "truth": 100, "pred": 100, "gate": 70, "ok": 60, "ms": 70}
        for c in cols:
            self.result_tree.heading(c, text=headings[c])
            self.result_tree.column(c, width=widths[c], anchor="center")
        self.result_tree.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.result_tree.yview)
        scroll.pack(side="right", fill="y")
        self.result_tree.configure(yscrollcommand=scroll.set)

    def _on_chart_resize(self, _event=None) -> None:
        if not self._chart_source_path or not self._chart_source_path.is_file():
            return
        if self._chart_resize_job is not None:
            try:
                self.after_cancel(self._chart_resize_job)
            except tk.TclError:
                pass
        self._chart_resize_job = self.after(
            80, lambda: self._fit_chart_to_panel(self._chart_source_path)
        )

    def _browse_analysis_run(self) -> None:
        initial = default_data_dir()
        initial.mkdir(parents=True, exist_ok=True)
        folder = filedialog.askdirectory(
            title="Select run folder (e.g. data/run1_20260812)",
            initialdir=str(initial),
        )
        if folder:
            self.analysis_run_var.set(folder)

    def _browse_sd(self) -> None:
        path = filedialog.askopenfilename(
            title="Select SD .csv (or cancel to pick a folder)",
            filetypes=[("CSV", "*.csv *.CSV"), ("All", "*.*")],
        )
        if path:
            self.sd_var.set(path)
            return
        folder = filedialog.askdirectory(title="Select SD card / captures folder")
        if folder:
            self.sd_var.set(folder)

    def _display_report(
        self,
        report,
        meta: RunMetadata,
        run_dir: Path,
        csv_path: Path,
        *,
        rewrite_results: bool = True,
    ) -> None:
        detail_path = results_path(run_dir)
        if rewrite_results:
            detail_path = write_report_csv(report, detail_path)

        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", "end")
        lines = [
            f"Run folder: {run_dir.name}",
            f"Run number: {meta.run_number or '?'}",
            "",
            f"OVERALL ACCURACY: {report.overall_accuracy * 100:.2f}%  "
            f"({report.correct}/{report.evaluated} evaluated; {report.skipped} skipped)",
            "",
        ]
        lines.extend(report.summary_lines())
        lines.append("")
        lines.append(f"Source CSV (SD inference): {csv_path}")
        lines.append(f"analysis_results.csv (paired table): {detail_path}")
        labels, matrix = confusion_matrix_labels(report)
        if labels:
            lines.append("")
            lines.append("Confusion (rows=truth, cols=pred):")
            lines.append("        " + " ".join(f"{l[:8]:>8}" for l in labels))
            for gt, row in zip(labels, matrix):
                lines.append(f"{gt[:8]:>8} " + " ".join(f"{v:8d}" for v in row))
        self.summary_text.insert("1.0", "\n".join(lines))
        self.summary_text.configure(state="disabled")

        for item in self.result_tree.get_children():
            self.result_tree.delete(item)
        for row in report.paired:
            self.result_tree.insert(
                "",
                "end",
                values=(
                    row.index,
                    row.ground_truth,
                    row.disease_pred,
                    row.gate_pred,
                    "yes" if row.correct else ("skip" if row.skipped else "no"),
                    row.infer_ms,
                ),
            )

        self._last_report_run_dir = run_dir
        self._render_chart(report, run_dir)

    def _run_analysis(self) -> None:
        run_raw = self.analysis_run_var.get().strip()
        sd_path = self.sd_var.get().strip()
        if not run_raw or not sd_path:
            messagebox.showwarning("Analysis", "Provide both a run folder and an SD log path.")
            return
        try:
            run_dir = resolve_run_dir(Path(run_raw))
            meta = load_run(run_dir)
            csv_path = find_sd_csv(Path(sd_path))
            preds = load_inference_csv(csv_path)
            report = analyze(meta, preds)
            archived = archive_log_to_run(csv_path, run_dir)
        except (OSError, ValueError, FileNotFoundError, KeyError) as exc:
            messagebox.showerror("Analysis", str(exc))
            return

        self._set_run(meta, run_dir, run_dir / "metadata.json")
        self.analysis_run_var.set(str(run_dir))
        self.sd_var.set(str(archived))
        self._display_report(report, meta, run_dir, archived)
        messagebox.showinfo(
            "Analysis complete",
            f"Overall accuracy: {report.overall_accuracy * 100:.2f}% "
            f"({report.correct}/{report.evaluated})\n"
            f"{archived.name} copied into {run_dir.name}\n"
            f"Saved analysis_results.csv + chart.png",
        )

    def _load_old_analysis(self) -> None:
        initial = default_data_dir()
        initial.mkdir(parents=True, exist_ok=True)
        suggested = self.analysis_run_var.get().strip()
        start = suggested if suggested and Path(suggested).is_dir() else str(initial)
        folder = filedialog.askdirectory(
            title="Select run folder to load analysis",
            initialdir=start,
        )
        if not folder:
            return
        try:
            run_dir = resolve_run_dir(Path(folder))
            meta = load_run(run_dir)
            report, csv_file, kind = load_saved_run_analysis(run_dir)
        except (OSError, ValueError, FileNotFoundError, KeyError) as exc:
            messagebox.showerror("Load analysis", str(exc))
            return

        self._set_run(meta, run_dir, run_dir / "metadata.json")
        self.analysis_run_var.set(str(run_dir))
        self.sd_var.set(str(csv_file))
        self._display_report(
            report,
            meta,
            run_dir,
            csv_file,
            rewrite_results=(kind == "sd_csv"),
        )
        source_note = (
            f"Recomputed from {csv_file.name}"
            if kind == "sd_csv"
            else f"Loaded from {csv_file.name}"
        )
        messagebox.showinfo(
            "Loaded analysis",
            f"Loaded {run_dir.name}\n"
            f"{source_note}\n"
            f"Accuracy: {report.overall_accuracy * 100:.2f}% "
            f"({report.correct}/{report.evaluated})",
        )


    def _fit_chart_to_panel(self, path: Path) -> None:
        path = Path(path)
        if not path.is_file():
            return
        self.chart_label.update_idletasks()
        max_w = max(self.chart_label.winfo_width() - 4, 640)
        max_h = max(self.chart_label.winfo_height() - 4, 320)
        with Image.open(path) as im:
            display = im.convert("RGB")
            src_w, src_h = display.size
            scale = min(max_w / src_w, max_h / src_h)
            new_size = (max(1, int(src_w * scale)), max(1, int(src_h * scale)))
            display = display.resize(new_size, Image.Resampling.LANCZOS)
            self._chart_photo = ImageTk.PhotoImage(display)
        self.chart_label.configure(image=self._chart_photo, text="")

    def _render_chart(self, report, run_dir: Path | None = None) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            from matplotlib.figure import Figure
            from matplotlib.colors import LinearSegmentedColormap
            import numpy as np
        except ImportError:
            self.chart_label.configure(image="", text="Install matplotlib to see charts.")
            return

        labels, _ = confusion_matrix_labels(report)
        if not labels and not report.per_class:
            self.chart_label.configure(image="", text="No evaluated samples.")
            return

        preferred = ["bacterial", "fungal", "healthy", "pest", "viral"]
        ordered = [c for c in preferred if c in labels]
        ordered.extend(c for c in labels if c not in ordered)
        labels = ordered
        matrix = [
            [int(report.confusion.get(gt, {}).get(pred, 0)) for pred in labels]
            for gt in labels
        ]

        fig = Figure(figsize=(14, 5.4), dpi=120, facecolor="white")
        gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.55, 1.7], wspace=0.32)
        ax_overall = fig.add_subplot(gs[0, 0])
        ax_cm = fig.add_subplot(gs[0, 1])
        ax_bar = fig.add_subplot(gs[0, 2])

        overall = report.overall_accuracy

        # ---- Overall accuracy (vertical bar, same style as per-class) ----
        overall_bar = ax_overall.bar(
            ["Overall"],
            [overall],
            color=ACCENT,
            edgecolor=ACCENT_DARK,
            width=0.55,
        )
        ax_overall.set_ylim(0, 1.05)
        ax_overall.set_ylabel("Accuracy")
        ax_overall.set_title("Overall accuracy", fontsize=12, fontweight="bold", color=ACCENT_DARK)
        good_line = ax_overall.axhline(
            0.70, color=ACCENT, linestyle=":", linewidth=1.4, label="Good (70%)"
        )
        ax_overall.tick_params(axis="x", labelsize=9)
        ax_overall.text(
            overall_bar[0].get_x() + overall_bar[0].get_width() / 2,
            min(overall + 0.03, 1.0),
            f"{overall * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
        ax_overall.text(
            0.5,
            -0.16,
            f"{report.correct}/{report.evaluated} evaluated · {report.skipped} skipped",
            ha="center",
            va="top",
            fontsize=8,
            color=MUTED,
            transform=ax_overall.transAxes,
        )
        ax_overall.legend(
            handles=[good_line],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            fontsize=8,
            frameon=False,
            ncol=1,
        )

        if labels:
            arr = np.array(matrix, dtype=float)
            cmap = LinearSegmentedColormap.from_list(
                "aatbs_blue", ["#f4f8fb", "#9ec5d8", "#2f6f8f", "#163b4d"]
            )
            im = ax_cm.imshow(arr, cmap=cmap, aspect="equal")
            cbar = fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.03)
            cbar.ax.tick_params(labelsize=8)
            ax_cm.set_xticks(range(len(labels)))
            ax_cm.set_yticks(range(len(labels)))
            ax_cm.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
            ax_cm.set_yticklabels(labels, fontsize=9)
            ax_cm.set_xlabel("Predicted class")
            ax_cm.set_ylabel("True class")
            ax_cm.set_title("Confusion matrix", fontsize=12, fontweight="bold")
            row_sums = arr.sum(axis=1)
            vmax = float(arr.max()) if arr.size else 1.0
            for i in range(len(labels)):
                for j in range(len(labels)):
                    count = int(arr[i, j])
                    if i == j and row_sums[i] > 0:
                        pct = 100.0 * count / row_sums[i]
                        text = f"{count}\n({pct:.1f}%)"
                    else:
                        text = str(count) if count else ""
                    color = "white" if count > vmax * 0.55 else TEXT
                    if text:
                        ax_cm.text(j, i, text, ha="center", va="center", fontsize=8, color=color)
        else:
            ax_cm.set_title("Confusion matrix (no data)")
            ax_cm.axis("off")

        classes = list(report.per_class.keys())
        bar_classes = [c for c in preferred if c in classes]
        bar_classes.extend(c for c in classes if c not in bar_classes)
        values = [report.per_class[c]["accuracy"] for c in bar_classes]
        if bar_classes:
            bars = ax_bar.bar(bar_classes, values, color=ACCENT, edgecolor=ACCENT_DARK, width=0.65)
            ax_bar.set_ylim(0, 1.05)
            ax_bar.set_ylabel("Accuracy")
            ax_bar.set_xlabel("Class")
            ax_bar.set_title("Per-class accuracy", fontsize=12, fontweight="bold")
            line_good = ax_bar.axhline(
                0.70, color=ACCENT, linestyle=":", linewidth=1.4, label="Good (70%)"
            )
            line_overall = ax_bar.axhline(
                overall,
                color="#b45309",
                linestyle="--",
                linewidth=1.3,
                label=f"Overall ({overall * 100:.1f}%)",
            )
            ax_bar.legend(
                handles=[line_good, line_overall],
                loc="upper center",
                bbox_to_anchor=(0.5, -0.22),
                fontsize=8,
                frameon=False,
                ncol=2,
            )
            ax_bar.tick_params(axis="x", rotation=20, labelsize=9)
            for bar, val in zip(bars, values):
                ax_bar.text(
                    bar.get_x() + bar.get_width() / 2,
                    min(val + 0.03, 1.0),
                    f"{val * 100:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        else:
            ax_bar.set_title("Per-class accuracy (no data)")
            ax_bar.axis("off")

        fig.subplots_adjust(left=0.05, right=0.98, top=0.90, bottom=0.24, wspace=0.30)

        target = run_dir or self._last_report_run_dir or default_data_dir()
        target.mkdir(parents=True, exist_ok=True)
        out_file = chart_path(target) if (run_dir or self._last_report_run_dir) else target / "chart.png"
        fig.savefig(out_file, facecolor="white")
        fig.clear()

        self._chart_source_path = out_file
        self._fit_chart_to_panel(out_file)



def run_app() -> None:
    app = AATBSApp()
    app.mainloop()
