import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import pandas as pd
import os
import threading
from disease_detector import DiseaseDetector

C = {
    'bg':           '#1a0533',    # deep violet-black
    'surface':      '#2a0a4a',    # rich purple
    'card':         '#3b1060',    # vivid dark purple
    'card_inner':   '#1a0533',    # deep violet-black
    'border':       '#7c3aed',    # violet-600
    'primary':      '#f97316',    # orange-500 (bold contrast)
    'primary_h':    '#ea580c',    # orange-600
    'success':      '#a3e635',    # lime-400 (electric green)
    'success_h':    '#84cc16',    # lime-500
    'warning':      '#fbbf24',    # amber-400
    'warning_h':    '#f59e0b',    # amber-500
    'danger':       '#fb7185',    # rose-400
    'accent':       '#e879f9',    # fuchsia-400 (neon pop)
    'accent_h':     '#d946ef',    # fuchsia-500
    'text':         '#fdf4ff',    # purple-50
    'text_muted':   '#d8b4fe',    # purple-300
    'text_dim':     '#7c3aed',    # violet-600
    'white':        '#ffffff',
}

FONT = "Segoe UI"

class CropDiseaseDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("Crop Disease Detection System")
        self.root.geometry("1280x820")
        self.root.configure(bg=C['bg'])
        self.root.resizable(True, True)
        self.root.minsize(1000, 700)

        self.image_paths = []          # list of selected image paths
        self.csv_path = None
        self.csv_data = None
        self.file_type = None
        self.detector = DiseaseDetector()
        self._thumb_refs = []            # keep PhotoImage refs alive

        self._build_ui()

    # ─── UI CONSTRUCTION ──────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_header()
        self._build_body()
        self._build_statusbar()

    def _build_header(self):
        hdr = tk.Frame(self.root, bg=C['surface'], height=80,
                       highlightthickness=1, highlightbackground=C['border'], highlightcolor=C['border'])
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        inner = tk.Frame(hdr, bg=C['surface'])
        inner.pack(expand=True, fill="both", padx=40)

        tk.Label(inner, text="🌱  CropGuard AI",
                 font=(FONT, 24, "bold"), bg=C['surface'], fg=C['text']).pack(side="left", pady=18)

        badge = tk.Label(inner, text="  ENTERPRISE  ",
                         font=(FONT, 8, "bold"), bg=C['success'], fg=C['white'],
                         padx=10, pady=2)
        badge.pack(side="left", padx=15, pady=28)

        tk.Label(inner, text="Intelligent Plant Disease Analysis",
                 font=(FONT, 10), bg=C['surface'], fg=C['text_muted']).pack(side="right", pady=28)

    def _build_body(self):
        body = tk.Frame(self.root, bg=C['bg'])
        body.pack(expand=True, fill="both", padx=32, pady=24)

        self._build_left(body)
        self._build_right(body)

    def _build_left(self, parent):
        left = tk.Frame(parent, bg=C['bg'])
        left.pack(side="left", fill="both", expand=True, padx=(0, 12))

        # Upload card
        upload_card = self._card(left, "📁  File Upload")
        upload_card.pack(fill="x", pady=(0, 14))

        btn_row = tk.Frame(upload_card, bg=C['card'])
        btn_row.pack(fill="x", padx=20, pady=(0, 18))

        self.img_btn = self._btn(btn_row, "📷  Upload Images", C['accent'], C['accent_h'], self.upload_image)
        self.img_btn.pack(side="left", padx=(0, 10))

        self.multi_btn = self._btn(btn_row, "🖼️  Multiple Image", C['primary'], C['primary_h'], self.upload_multiple_images)
        self.multi_btn.pack(side="left", padx=(0, 10))

        self.add_btn = self._btn(btn_row, "➕  Add More", C['accent'], C['accent_h'], self.add_images)
        self.add_btn.pack(side="left", padx=(0, 10))
        self.add_btn.pack_forget()

        self.csv_btn = self._btn(btn_row, "📊  Upload CSV", C['warning'], C['warning_h'], self.upload_csv)
        self.csv_btn.pack(side="left")

        # Preview card
        prev_card = self._card(left, "🖼️  Preview")
        prev_card.pack(fill="both", expand=True)

        self.display_frame = tk.Frame(prev_card, bg=C['surface'],
                                      highlightthickness=1, highlightbackground=C['border'])
        self.display_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        self.display_label = tk.Label(self.display_frame,
                                      text="📸\n\nWaiting for Data\n\nPlease upload a crop image or a CSV dataset\nto begin the automated analysis.",
                                      font=(FONT, 12), bg=C['surface'], fg=C['text_dim'],
                                      justify="center", pady=40)
        self.display_label.pack(expand=True)

    def _build_right(self, parent):
        right = tk.Frame(parent, bg=C['bg'], width=420)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        # Analyse card
        ctrl_card = self._card(right, "🔬  Analysis")
        ctrl_card.pack(fill="x", pady=(0, 14))

        self.detect_btn = self._btn(ctrl_card, "🚀  START ANALYSIS", C['success'], C['success_h'],
                                    self.detect_disease, big=True)
        self.detect_btn.pack(fill="x", padx=20, pady=(0, 20))
        self.detect_btn.config(state="disabled", bg=C['text_dim'], activebackground=C['text_dim'])

        # Progress bar (hidden until analysis)
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(ctrl_card, variable=self.progress_var,
                                        maximum=100, mode="indeterminate", length=200)
        self.progress.pack(fill="x", padx=20, pady=(0, 14))
        self.progress.pack_forget()

        # Results card
        res_card = self._card(right, "📊  Results")
        res_card.pack(fill="both", expand=True)

        res_inner = tk.Frame(res_card, bg=C['card'])
        res_inner.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        # Disease row
        self._label(res_inner, "DISEASE DETECTED", C['text_dim'], 8).pack(anchor="w", pady=(0, 4))
        self.disease_label = tk.Label(res_inner, text="—",
                                      font=(FONT, 16, "bold"), bg=C['card'], fg=C['text_muted'])
        self.disease_label.pack(anchor="w", pady=(0, 14))

        # Confidence row
        self._label(res_inner, "CONFIDENCE", C['text_dim'], 8).pack(anchor="w", pady=(0, 4))
        conf_row = tk.Frame(res_inner, bg=C['card'])
        conf_row.pack(fill="x", pady=(0, 14))

        self.conf_badge = tk.Label(conf_row, text="0%",
                                   font=(FONT, 14, "bold"), bg=C['text_dim'], fg=C['white'],
                                   padx=12, pady=4)
        self.conf_badge.pack(side="left")

        self.conf_bar_bg = tk.Frame(conf_row, bg=C['border'], height=8)
        self.conf_bar_bg.pack(side="left", fill="x", expand=True, padx=(10, 0), pady=8)
        self.conf_bar = tk.Frame(self.conf_bar_bg, bg=C['text_dim'], height=8, width=0)
        self.conf_bar.place(x=0, y=0, relheight=1)

        # Treatment
        self._label(res_inner, "TREATMENT RECOMMENDATION", C['text_dim'], 8).pack(anchor="w", pady=(0, 6))

        txt_frame = tk.Frame(res_inner, bg=C['surface'],
                             highlightthickness=1, highlightbackground=C['border'])
        txt_frame.pack(fill="both", expand=True)

        self.remedy_text = tk.Text(txt_frame, wrap="word", font=(FONT, 10),
                                   bg=C['surface'], fg=C['text_muted'],
                                   relief="flat", bd=0, padx=12, pady=10,
                                   insertbackground=C['text'])
        sb = tk.Scrollbar(txt_frame, orient="vertical", command=self.remedy_text.yview,
                          bg=C['surface'], troughcolor=C['surface'])
        self.remedy_text.configure(yscrollcommand=sb.set)
        self.remedy_text.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        self._set_remedy("Upload a file and run analysis to receive treatment recommendations.")

    def _build_statusbar(self):
        bar = tk.Frame(self.root, bg=C['surface'], height=32,
                       highlightthickness=1, highlightbackground=C['border'])
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        inner = tk.Frame(bar, bg=C['surface'])
        inner.pack(fill="both", expand=True, padx=20)

        self.status_var = tk.StringVar(value="● SYSTEM READY")
        tk.Label(inner, textvariable=self.status_var,
                 font=(FONT, 9, "bold"), bg=C['surface'], fg=C['success']).pack(side="left")

        tk.Label(inner, text="v1.2.4-STABLE",
                 font=(FONT, 8, "bold"), bg=C['surface'], fg=C['text_dim']).pack(side="right")

    # ─── HELPERS ──────────────────────────────────────────────────────────────

    def _card(self, parent, title):
        frame = tk.Frame(parent, bg=C['card'],
                         highlightthickness=1, highlightbackground=C['border'])
        
        # Header for card
        header = tk.Frame(frame, bg=C['card'])
        header.pack(fill="x", padx=20, pady=(18, 14))
        
        indicator = tk.Frame(header, bg=C['accent'], width=4, height=18)
        indicator.pack(side="left")
        indicator.pack_propagate(False)

        tk.Label(header, text=title, font=(FONT, 12, "bold"),
                 bg=C['card'], fg=C['text']).pack(side="left", padx=10)
        
        return frame

    def _btn(self, parent, text, color, hover, cmd, big=False):
        size = 12 if big else 10
        pad_y = 14 if big else 10
        b = tk.Button(parent, text=text, command=cmd,
                      font=(FONT, size, "bold"), bg=color, fg=C['white'],
                      relief="flat", padx=24, pady=pad_y, cursor="hand2",
                      activebackground=hover, activeforeground=C['white'],
                      disabledforeground=C['text_dim'])
        b.bind("<Enter>", lambda e: b.config(bg=hover) if b['state'] == 'normal' else None)
        b.bind("<Leave>", lambda e: b.config(bg=color) if b['state'] == 'normal' else None)
        b._color = color
        return b

    def _label(self, parent, text, color, size):
        return tk.Label(parent, text=text.upper(), font=(FONT, size, "bold"),
                        bg=C['card'], fg=color, pady=2)

    def _set_remedy(self, text):
        self.remedy_text.config(state="normal")
        self.remedy_text.delete(1.0, tk.END)
        self.remedy_text.insert(1.0, text)
        self.remedy_text.config(state="disabled")

    def _set_status(self, msg):
        self.status_var.set(msg)

    def _update_conf_bar(self, pct, color):
        self.conf_bar_bg.update_idletasks()
        total = self.conf_bar_bg.winfo_width()
        self.conf_bar.place(x=0, y=0, relheight=1, width=int(total * pct / 100))
        self.conf_bar.config(bg=color)

    # ─── FILE UPLOAD ──────────────────────────────────────────────────────────

    def upload_image(self):
        paths = filedialog.askopenfilenames(
            title="Select Crop Images",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )
        if not paths:
            return
        self.image_paths = list(paths)
        self.csv_path, self.file_type = None, 'image'
        self._display_image_grid()
        self._enable_detect()
        self.img_btn.config(text="🔄  Replace Images", bg=C['accent'])
        self.add_btn.pack(side="left", padx=(0, 10))
        self.csv_btn.config(text="📊  Upload CSV", bg=C['warning'])
        self._reset_results()
        self._set_status(f"Loaded {len(self.image_paths)} image(s)")

    def add_images(self):
        paths = filedialog.askopenfilenames(
            title="Add More Crop Images",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )
        if not paths:
            return
        new = [p for p in paths if p not in self.image_paths]
        self.image_paths.extend(new)
        self._display_image_grid()
        self._reset_results()
        self._set_status(f"{len(self.image_paths)} image(s) loaded  (+{len(new)} added)")

    def upload_multiple_images(self):
        paths = filedialog.askopenfilenames(
            title="Select Multiple Crop Images",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )
        if not paths:
            return
        self.image_paths = list(paths)
        self.csv_path, self.file_type = None, 'image'
        self._display_image_grid()
        self._enable_detect()
        self.multi_btn.config(text=f"✅  {len(self.image_paths)} Images", bg=C['success'])
        self.img_btn.config(text="📷  Upload Images", bg=C['accent'])
        self.add_btn.pack(side="left", padx=(0, 10))
        self.csv_btn.config(text="📊  Upload CSV", bg=C['warning'])
        self._reset_results()
        self._set_status(f"Loaded {len(self.image_paths)} image(s) via Multiple Image")

    def upload_csv(self):
        path = filedialog.askopenfilename(
            title="Select CSV / Excel File",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            self.csv_data = pd.read_excel(path) if path.endswith(('.xlsx', '.xls')) else pd.read_csv(path)
            self.csv_path, self.image_paths, self.file_type = path, [], 'csv'
            self._display_csv(path)
            self._enable_detect()
            self.csv_btn.config(text="✅  CSV Loaded", bg=C['success'])
            self.img_btn.config(text="📷  Upload Images", bg=C['accent'])
            self.multi_btn.config(text="🖼️  Multiple Image", bg=C['primary'])
            self.add_btn.pack_forget()
            self._reset_results()
            self._set_status(f"Loaded: {os.path.basename(path)}  ({self.csv_data.shape[0]} rows)")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{e}")

    def _enable_detect(self):
        self.detect_btn.config(state="normal", bg=C['success'],
                               activebackground=C['success_h'])

    # ─── DISPLAY ──────────────────────────────────────────────────────────────

    def _clear_display(self):
        for w in self.display_frame.winfo_children():
            w.destroy()

    def _display_image_grid(self):
        """Render all images in a scrollable grid, each with a ✕ close button."""
        self._clear_display()
        self._thumb_refs.clear()

        canvas = tk.Canvas(self.display_frame, bg=C['surface'], highlightthickness=0)
        sb = tk.Scrollbar(self.display_frame, orient="vertical", command=canvas.yview,
                          bg=C['surface'], troughcolor=C['surface'])
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas, bg=C['surface'])
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_resize(e):
            canvas.itemconfig(win_id, width=e.width)
        canvas.bind("<Configure>", _on_resize)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(-1*(e.delta//120), "units"))

        COLS = 3
        for idx, path in enumerate(self.image_paths):
            row, col = divmod(idx, COLS)
            cell = tk.Frame(inner, bg=C['card'], padx=10, pady=10,
                            highlightthickness=1, highlightbackground=C['border'])
            cell.grid(row=row, column=col, padx=8, pady=8, sticky="nsew")
            inner.columnconfigure(col, weight=1)

            try:
                img = Image.open(path)
                img.thumbnail((140, 110), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self._thumb_refs.append(photo)
                tk.Label(cell, image=photo, bg=C['card']).pack(pady=(6, 2))
            except Exception:
                tk.Label(cell, text="⚠️ Error", bg=C['card'], fg=C['danger'],
                         font=(FONT, 9)).pack(pady=(6, 2))

            tk.Label(cell, text=os.path.basename(path), font=(FONT, 7),
                     bg=C['card'], fg=C['text_muted'], wraplength=130).pack()

            close_btn = tk.Button(
                cell, text="✕", font=(FONT, 8, "bold"),
                bg=C['danger'], fg=C['white'], relief="flat",
                padx=4, pady=1, cursor="hand2",
                activebackground=C['warning_h'], activeforeground=C['white'],
                command=lambda p=path: self._remove_image(p)
            )
            close_btn.pack(pady=(2, 6))

    def _remove_image(self, path):
        self.image_paths.remove(path)
        if not self.image_paths:
            self._clear_display()
            self.display_label = tk.Label(
                self.display_frame,
                text="📷\n\nNo file selected\nUpload an image or CSV to begin",
                font=(FONT, 11), bg=C['surface'], fg=C['text_dim'], justify="center")
            self.display_label.pack(expand=True)
            self.detect_btn.config(state="disabled", bg=C['text_dim'],
                                   activebackground=C['text_dim'])
            self.img_btn.config(text="📷  Upload Images", bg=C['accent'])
            self.multi_btn.config(text="🖼️  Multiple Image", bg=C['primary'])
            self.add_btn.pack_forget()
            self._reset_results()
            self._set_status("Ready")
        else:
            self.img_btn.config(text="🔄  Replace Images", bg=C['accent'])
            self._set_status(f"{len(self.image_paths)} image(s) remaining")
            self._display_image_grid()

    def _display_image_obj(self, img_obj, label_text=""):
        img_obj.thumbnail((480, 340), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img_obj)
        self._clear_display()
        lbl = tk.Label(self.display_frame, image=photo, bg=C['surface'])
        lbl.pack(expand=True, pady=(12, 4))
        lbl.image = photo
        tk.Label(self.display_frame, text=label_text,
                 font=(FONT, 9), bg=C['surface'], fg=C['text_muted']).pack(pady=(0, 10))

    def _display_csv(self, path):
        self._clear_display()
        rows, cols = self.csv_data.shape
        f = tk.Frame(self.display_frame, bg=C['surface'])
        f.pack(expand=True)

        tk.Label(f, text="📊", font=(FONT, 36), bg=C['surface'], fg=C['accent']).pack(pady=(20, 8))
        tk.Label(f, text=os.path.basename(path),
                 font=(FONT, 13, "bold"), bg=C['surface'], fg=C['text']).pack()
        tk.Label(f, text=f"{rows:,} rows  ×  {cols} columns",
                 font=(FONT, 10), bg=C['surface'], fg=C['text_muted']).pack(pady=6)

        cols_preview = ", ".join(self.csv_data.columns[:6].tolist())
        if len(self.csv_data.columns) > 6:
            cols_preview += f"  … +{len(self.csv_data.columns)-6} more"
        tk.Label(f, text=cols_preview, font=(FONT, 8), bg=C['surface'],
                 fg=C['text_dim'], wraplength=380).pack(pady=(4, 20))

    def _display_csv_table(self, results):
        self._clear_display()
        # Table wrapper
        table_wrap = tk.Frame(self.display_frame, bg=C['surface'], padx=10, pady=10)
        table_wrap.pack(fill="both", expand=True)

        # Header
        hdr = tk.Frame(table_wrap, bg=C['surface'])
        hdr.pack(fill="x", pady=(0, 4))
        for text, w in [("INDEX", 8), ("DIAGNOSIS", 22), ("CONFIDENCE", 15), ("STATUS", 10)]:
            tk.Label(hdr, text=text, font=(FONT, 9, "bold"), bg=C['surface'], fg=C['text_muted'],
                     width=w, anchor="w", padx=10, pady=8).pack(side="left")

        # Scrollable area
        container = tk.Frame(table_wrap, bg=C['bg'], highlightthickness=1, highlightbackground=C['border'])
        container.pack(fill="both", expand=True)

        canvas = tk.Canvas(container, bg=C['bg'], highlightthickness=0)
        sb = tk.Scrollbar(table_wrap, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=C['bg'])
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        
        def _on_canvas_configure(e):
            canvas.itemconfig(1, width=e.width)
        canvas.bind("<Configure>", _on_canvas_configure)

        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        disease_colors = {
            "Healthy": C['success'],
            "Leaf Blight": C['danger'],
            "Powdery Mildew": C['warning'],
            "Rust Disease": C['warning_h'],
        }
        for i, r in enumerate(results):
            row_bg = C['surface'] if i % 2 == 0 else C['bg']
            color = disease_colors.get(r["disease"], C['text_muted'])
            row_frame = tk.Frame(inner, bg=row_bg)
            row_frame.pack(fill="x")
            
            idx_str = f"{i + 1:02}"
            conf_str = f"{r['confidence']}%"
            status_icon = "✅" if r["disease"] == "Healthy" else "⚠️"
            
            tk.Label(row_frame, text=idx_str, font=(FONT, 9, "bold"), bg=row_bg,
                     fg=C['text_dim'], width=8, anchor="w", padx=10, pady=6).pack(side="left")
            
            tk.Label(row_frame, text=r["disease"].upper(), font=(FONT, 9, "bold"), bg=row_bg,
                     fg=color, width=22, anchor="w", padx=10).pack(side="left")
            
            tk.Label(row_frame, text=conf_str, font=(FONT, 9), bg=row_bg,
                     fg=C['text'], width=15, anchor="w", padx=10).pack(side="left")
            
            tk.Label(row_frame, text=status_icon, font=(FONT, 10), bg=row_bg,
                     fg=color, width=10, anchor="w", padx=10).pack(side="left")

    # ─── ANALYSIS ─────────────────────────────────────────────────────────────

    def detect_disease(self):
        self.detect_btn.config(state="disabled", bg=C['text_dim'])
        self.progress.pack(fill="x", padx=20, pady=(0, 14))
        self.progress.start(12)
        self._set_status("Analysing…")
        threading.Thread(target=self._run_analysis, daemon=True).start()

    def _run_analysis(self):
        try:
            if self.file_type == 'image':
                results = []
                non_crop = []
                for path in self.image_paths:
                    try:
                        disease, conf, remedy, marked = self.detector.analyze_image(path)
                        results.append((path, disease, conf, remedy, marked))
                    except ValueError:
                        non_crop.append(os.path.basename(path))
                if non_crop and not results:
                    self.root.after(0, lambda n=non_crop: self._analysis_error(
                        "Not a crop image:\n" + "\n".join(n) +
                        "\n\nPlease upload images of crop or plant leaves."))
                    return
                if non_crop:
                    self.root.after(0, lambda n=non_crop: messagebox.showwarning(
                        "Non-Crop Images Skipped",
                        "The following are not crop images and were skipped:\n" + "\n".join(n)))
                self.root.after(0, lambda r=results: self._finish_multi_image(r))
            else:
                disease, conf, remedy, rows = self.detector.analyze_csv(self.csv_data)
                self.root.after(0, lambda: self._finish_csv(disease, conf, remedy, rows))
        except Exception as e:
            self.root.after(0, lambda: self._analysis_error(str(e)))

    def _finish_multi_image(self, results):
        """Single image → large preview UI. Multiple images → scrollable grid with per-image View Result button."""
        if len(results) == 1:
            path, disease, conf, remedy, marked = results[0]
            self._finish_image(disease, conf, remedy, marked)
            return

        self._clear_display()
        self._thumb_refs.clear()

        canvas = tk.Canvas(self.display_frame, bg=C['surface'], highlightthickness=0)
        sb = tk.Scrollbar(self.display_frame, orient="vertical", command=canvas.yview,
                          bg=C['surface'], troughcolor=C['surface'])
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas, bg=C['surface'])
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win_id, width=e.width))
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(-1*(e.delta//120), "units"))

        COLS = 3
        disease_colors = {
            "Healthy": C['success'], "Leaf Blight": C['danger'],
            "Powdery Mildew": C['warning'], "Rust Disease": C['warning_h'],
        }
        for idx, (path, disease, conf, remedy, marked) in enumerate(results):
            row, col = divmod(idx, COLS)
            color = disease_colors.get(disease, C['text_muted'])
            cell = tk.Frame(inner, bg=C['card'], padx=10, pady=10,
                            highlightthickness=1, highlightbackground=color)
            cell.grid(row=row, column=col, padx=8, pady=8, sticky="nsew")
            inner.columnconfigure(col, weight=1)

            thumb = marked.copy()
            thumb.thumbnail((140, 110), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(thumb)
            self._thumb_refs.append(photo)
            tk.Label(cell, image=photo, bg=C['card']).pack(pady=(6, 2))

            icon = "✅" if disease == "Healthy" else "⚠️"
            tk.Label(cell, text=f"{icon} {disease}", font=(FONT, 8, "bold"),
                     bg=C['card'], fg=color).pack()
            tk.Label(cell, text=f"{conf}% confidence", font=(FONT, 7),
                     bg=C['card'], fg=C['text_muted']).pack()
            tk.Label(cell, text=os.path.basename(path), font=(FONT, 7),
                     bg=C['card'], fg=C['text_dim'], wraplength=130).pack(pady=(0, 4))

            view_btn = tk.Button(
                cell, text="📋  View Result",
                font=(FONT, 7, "bold"), bg=C['primary'], fg=C['white'],
                relief="flat", padx=6, pady=3, cursor="hand2",
                activebackground=C['primary_h'], activeforeground=C['white'],
                command=lambda d=disease, c=conf, r=remedy: self._display_results(d, c, r)
            )
            view_btn.pack(pady=(0, 6))

        # Show first result in right panel by default
        self._finish(*results[0][1:4])

    def _finish_image(self, disease, conf, remedy, marked):
        self._display_image_obj(marked, f"🔍  Analysis complete")
        self._finish(disease, conf, remedy)

    def _finish_csv(self, disease, conf, remedy, rows):
        self._display_csv_table(rows)
        self._finish(disease, conf, remedy)

    def _finish(self, disease, conf, remedy):
        self.progress.stop()
        self.progress.pack_forget()
        self.detect_btn.config(state="normal", bg=C['success'], activebackground=C['success_h'])
        self._display_results(disease, conf, remedy)
        self._set_status(f"Analysis complete — {disease}  ({conf}% confidence)")

    def _analysis_error(self, msg):
        self.progress.stop()
        self.progress.pack_forget()
        self.detect_btn.config(state="normal", bg=C['success'], activebackground=C['success_h'])
        self._set_status("Analysis failed")
        messagebox.showerror("Analysis Error", msg)

    # ─── RESULTS ──────────────────────────────────────────────────────────────

    def _display_results(self, disease, conf, remedy):
        if disease == "Healthy":
            color = C['success']
            icon = "✅"
            label = "HEALTHY"
            status_text = f"ANALYSIS REPORT: STABLE HEALTH\n" + "─" * 40 + "\n\n"
            status_text += f"Confidence Score: {conf}%\n\n"
            status_text += "MAINTENANCE RECOMMENDATIONS:\n" + remedy
        else:
            color = C['danger'] if conf >= 80 else C['warning']
            icon = "⚠️"
            label = disease.upper()
            status_text = f"ANALYSIS REPORT: PATHOGEN DETECTED\n" + "─" * 40 + "\n\n"
            status_text += f"Identity: {disease}\n"
            status_text += f"Confidence Score: {conf}%\n\n"
            status_text += "TREATMENT PROTOCOL:\n" + remedy

        self.disease_label.config(text=f"{icon}  {label}", fg=color)
        self.conf_badge.config(text=f"{conf}%", bg=color)
        self.root.after(50, lambda: self._update_conf_bar(conf, color))
        self._set_remedy(status_text)

    def _reset_results(self):
        self.disease_label.config(text="—", fg=C['text_muted'])
        self.conf_badge.config(text="0%", bg=C['text_dim'])
        self.conf_bar.place(x=0, y=0, relheight=1, width=0)
        self._set_remedy("System ready for analysis.\nUpload a file to receive diagnostic reports and treatment protocols.")


def main():
    root = tk.Tk()
    w, h = 1280, 820
    x = (root.winfo_screenwidth() - w) // 2
    y = (root.winfo_screenheight() - h) // 2
    root.geometry(f"{w}x{h}+{x}+{y}")

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("Horizontal.TProgressbar",
                    troughcolor=C['bg'], background=C['primary'],
                    thickness=8, borderwidth=0)
    # TNotebook/Treeview styles if needed
    style.configure("Treeview", background=C['surface'], foreground=C['text'], fieldbackground=C['surface'], borderwidth=0)
    style.map("Treeview", background=[('selected', C['primary'])])

    CropDiseaseDetector(root)
    root.mainloop()

if __name__ == "__main__":
    main()
