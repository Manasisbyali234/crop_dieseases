import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import pandas as pd
import os
import threading
from disease_detector import DiseaseDetector

C = {
    'bg':           '#f1f5f9',    # slate-100
    'surface':      '#ffffff',    # white
    'card':         '#ffffff',    # white
    'card_inner':   '#f8fafc',    # slate-50
    'border':       '#cbd5e1',    # slate-300
    'border_glow':  '#10b98133',
    'primary':      '#059669',    # emerald-600
    'primary_h':    '#10b981',    # emerald-500
    'primary_dim':  '#d1fae5',    # emerald-100
    'success':      '#059669',
    'success_h':    '#10b981',
    'warning':      '#d97706',    # amber-600
    'warning_h':    '#f59e0b',    # amber-500
    'danger':       '#e11d48',    # rose-600
    'danger_dim':   '#ffe4e6',    # rose-100
    'accent':       '#4f46e5',    # indigo-600
    'accent_h':     '#6366f1',    # indigo-500
    'accent_dim':   '#e0e7ff',    # indigo-100
    'text':         '#0f172a',    # slate-900
    'text_muted':   '#475569',    # slate-600
    'text_dim':     '#94a3b8',    # slate-400
    'white':        '#ffffff',
    'header_top':   '#ffffff',
    'header_bot':   '#f1f5f9',
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
        try:
            self.root.tk.call('tk', 'scaling', 1.25)
        except Exception:
            pass

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
        hdr = tk.Canvas(self.root, height=72, bg=C['header_top'], highlightthickness=0)
        hdr.pack(fill="x")

        # Gradient simulation via two overlapping frames
        hdr.bind("<Configure>", lambda e: (
            hdr.delete("grad"),
            hdr.create_rectangle(0, 0, e.width, 72, fill=C['header_top'], outline="", tags="grad"),
            hdr.create_rectangle(0, 50, e.width, 72, fill=C['header_bot'], outline="", tags="grad"),
            hdr.create_line(0, 71, e.width, 71, fill=C['border'], width=1, tags="grad"),
            hdr.lower("grad")
        ))

        inner = tk.Frame(hdr, bg=C['header_top'])
        hdr.create_window(0, 0, window=inner, anchor="nw", tags="content")
        hdr.bind("<Configure>", lambda e, i=inner: (
            hdr.delete("grad"),
            hdr.create_rectangle(0, 0, e.width, 72, fill=C['header_top'], outline="", tags="grad"),
            hdr.create_rectangle(0, 50, e.width, 72, fill=C['header_bot'], outline="", tags="grad"),
            hdr.create_line(0, 71, e.width, 71, fill=C['border'], width=1, tags="grad"),
            hdr.lower("grad"),
            hdr.itemconfig("content", width=e.width)
        ))
        inner.configure(height=72)
        inner.pack_propagate(False)

        left_grp = tk.Frame(inner, bg=C['header_top'])
        left_grp.pack(side="left", padx=36, pady=0, fill="y")

        # Icon + brand
        brand_row = tk.Frame(left_grp, bg=C['header_top'])
        brand_row.pack(side="left", fill="y")
        tk.Label(brand_row, text="🌿", font=(FONT, 22), bg=C['header_top']).pack(side="left", padx=(0, 10))
        tk.Label(brand_row, text="CropGuard", font=(FONT, 20, "bold"),
                 bg=C['header_top'], fg=C['text']).pack(side="left")
        tk.Label(brand_row, text=" AI", font=(FONT, 20, "bold"),
                 bg=C['header_top'], fg=C['primary']).pack(side="left")

        # Version pill badge
        badge = tk.Frame(brand_row, bg=C['primary_dim'],
                         highlightthickness=1, highlightbackground=C['primary'])
        badge.pack(side="left", padx=(12, 0))
        tk.Label(badge, text="v2.0", font=(FONT, 7, "bold"),
                 bg=C['primary_dim'], fg=C['primary'], padx=8, pady=2).pack()

        # Right side — subtitle + live dot
        right_grp = tk.Frame(inner, bg=C['header_top'])
        right_grp.pack(side="right", padx=36, fill="y")
        tk.Label(right_grp, text="Advanced Plant Pathology Engine",
                 font=(FONT, 9), bg=C['header_top'], fg=C['text_dim']).pack(side="right", pady=(28, 0))

    def _build_body(self):
        body = tk.Frame(self.root, bg=C['bg'])
        body.pack(expand=True, fill="both", padx=30, pady=20)

        self._build_left(body)
        self._build_right(body)

    def _build_left(self, parent):
        left = tk.Frame(parent, bg=C['bg'])
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # Upload card
        upload_card = self._card(left, "Data Acquisition")
        upload_card.pack(fill="x", pady=(0, 14))

        btn_row = tk.Frame(upload_card, bg=C['card'])
        btn_row.pack(fill="x", padx=20, pady=(0, 18))

        self.img_btn = self._btn(btn_row, "🖼  Single Image", C['accent'], C['accent_h'], self.upload_images)
        self.img_btn.pack(side="left", padx=(0, 10))

        self.multi_btn = self._btn(btn_row, "🗂  Multiple Images", C['primary'], C['primary_h'], self.upload_multiple_images)
        self.multi_btn.pack(side="left", padx=(0, 10))

        self.add_btn = self._btn(btn_row, "＋ Add More", C['accent'], C['accent_h'], self.add_images)
        self.add_btn.pack(side="left", padx=(0, 10))
        self.add_btn.pack_forget()

        self.csv_btn = self._btn(btn_row, "📊 Dataset (CSV)", C['warning'], C['warning_h'], self.upload_csv)
        self.csv_btn.pack(side="left")

        # Image count badge
        self.count_label = tk.Label(btn_row, text="", font=(FONT, 8, "bold"),
                                    bg=C['card'], fg=C['text_dim'])
        self.count_label.pack(side="left", padx=(14, 0))

        # Preview card
        prev_card = self._card(left, "Visual Data Stream")
        prev_card.pack(fill="both", expand=True)

        self.display_frame = tk.Frame(prev_card, bg=C['surface'],
                                      highlightthickness=1, highlightbackground=C['border'])
        self.display_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        self.display_label = tk.Label(self.display_frame,
                                      text="SYSTEM IDLE\n\nAwaiting input stream for analysis.\nPlease provide agricultural imagery or a CSV dataset\nto initialize the diagnostic engine.",
                                      font=(FONT, 10), bg=C['surface'], fg=C['text_dim'],
                                      justify="center", pady=40)
        self.display_label.pack(expand=True)

    def _build_right(self, parent):
        right = tk.Frame(parent, bg=C['bg'], width=400)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        # Analyse card
        ctrl_card = self._card(right, "Diagnostic Controls", C['accent'])
        ctrl_card.pack(fill="x", pady=(0, 14))

        self.detect_btn = self._btn(ctrl_card, "EXECUTE DIAGNOSIS", C['primary'], C['primary_h'],
                                    self.detect_disease, big=True)
        self.detect_btn.pack(fill="x", padx=20, pady=(0, 20))
        self.detect_btn.config(state="disabled", bg=C['text_dim'])

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(ctrl_card, variable=self.progress_var,
                                        maximum=100, mode="indeterminate", length=200)
        self.progress.pack(fill="x", padx=20, pady=(0, 20))
        self.progress.pack_forget()

        # Results card
        res_card = self._card(right, "Clinical Assessment", C['warning'])
        res_card.pack(fill="both", expand=True)

        res_inner = tk.Frame(res_card, bg=C['card'])
        res_inner.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        # Disease row
        tk.Label(res_inner, text="IDENTIFIED CONDITION", font=(FONT, 7, "bold"),
                 bg=C['card'], fg=C['text_dim']).pack(anchor="w", pady=(0, 4))
        self.disease_label = tk.Label(res_inner, text="PENDING",
                                      font=(FONT, 16, "bold"), bg=C['card'], fg=C['text_muted'])
        self.disease_label.pack(anchor="w", pady=(0, 16))

        # Confidence row
        tk.Label(res_inner, text="CONFIDENCE LEVEL", font=(FONT, 7, "bold"),
                 bg=C['card'], fg=C['text_dim']).pack(anchor="w", pady=(0, 6))

        conf_container = tk.Frame(res_inner, bg=C['card_inner'], padx=12, pady=12,
                                  highlightthickness=1, highlightbackground=C['border'])
        conf_container.pack(fill="x", pady=(0, 16))

        self.conf_badge = tk.Label(conf_container, text="0%",
                                   font=(FONT, 14, "bold"), bg=C['card_inner'], fg=C['primary'])
        self.conf_badge.pack(side="left")

        bar_col = tk.Frame(conf_container, bg=C['card_inner'])
        bar_col.pack(side="left", fill="x", expand=True, padx=(14, 0))

        self.conf_bar_bg = tk.Canvas(bar_col, bg=C['card_inner'], height=10,
                                     highlightthickness=0)
        self.conf_bar_bg.pack(fill="x", pady=(4, 0))
        self.conf_bar_bg.bind("<Configure>", lambda e: self._redraw_conf_bar())
        self._conf_pct = 0
        self._conf_color = C['primary']

        # Remedy
        tk.Label(res_inner, text="REHABILITATION STRATEGY", font=(FONT, 7, "bold"),
                 bg=C['card'], fg=C['text_dim']).pack(anchor="w", pady=(0, 6))

        txt_frame = tk.Frame(res_inner, bg=C['card_inner'],
                             highlightthickness=1, highlightbackground=C['border'])
        txt_frame.pack(fill="both", expand=True)

        self.remedy_text = tk.Text(txt_frame, wrap="word", font=(FONT, 10),
                                   bg=C['card_inner'], fg=C['text_muted'],
                                   relief="flat", bd=0, padx=12, pady=12,
                                   insertbackground=C['text'], spacing2=5)
        sb = tk.Scrollbar(txt_frame, orient="vertical", command=self.remedy_text.yview)
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
        inner.pack(fill="both", expand=True, padx=28)

        # Animated live dot
        self._dot_canvas = tk.Canvas(inner, width=10, height=10,
                                     bg=C['surface'], highlightthickness=0)
        self._dot_canvas.pack(side="left", pady=11)
        self._dot_id = self._dot_canvas.create_oval(2, 2, 9, 9, fill=C['success'], outline="")
        self._dot_state = True
        self._animate_dot()

        self.status_var = tk.StringVar(value="READY")
        tk.Label(inner, text="STATUS", font=(FONT, 7, "bold"),
                 bg=C['surface'], fg=C['text_dim']).pack(side="left", padx=(6, 4))
        tk.Label(inner, textvariable=self.status_var,
                 font=(FONT, 7, "bold"), bg=C['surface'], fg=C['success']).pack(side="left")

        # Right side info
        tk.Label(inner, text="CROPGUARD AI  •  PLANT PATHOLOGY ENGINE",
                 font=(FONT, 7), bg=C['surface'], fg=C['text_dim']).pack(side="right")

    def _animate_dot(self):
        if not self.root.winfo_exists():
            return
        self._dot_state = not self._dot_state
        color = C['success'] if self._dot_state else C['primary_dim']
        self._dot_canvas.itemconfig(self._dot_id, fill=color)
        self.root.after(900, self._animate_dot)

    # ─── HELPERS ──────────────────────────────────────────────────────────────

    def _card(self, parent, title, accent=None):
        accent = accent or C['primary']
        frame = tk.Frame(parent, bg=C['card'],
                         highlightthickness=1, highlightbackground=C['border'])

        header = tk.Frame(frame, bg=C['card'])
        header.pack(fill="x", padx=18, pady=(16, 12))

        # Glowing left bar
        bar_canvas = tk.Canvas(header, width=4, height=22, bg=C['card'], highlightthickness=0)
        bar_canvas.pack(side="left")
        bar_canvas.create_rectangle(0, 0, 4, 22, fill=accent, outline="")
        bar_canvas.create_rectangle(0, 0, 4, 22, fill=accent, outline="", stipple="gray50")

        tk.Label(header, text=title.upper(), font=(FONT, 9, "bold"),
                 bg=C['card'], fg=C['text_muted']).pack(side="left", padx=10)

        sep = tk.Frame(frame, bg=C['border'], height=1)
        sep.pack(fill="x", padx=18, pady=(0, 16))

        return frame

    def _btn(self, parent, text, color, hover, cmd, big=False):
        size = 11 if big else 9
        pad_y = 13 if big else 8
        pad_x = 24 if big else 18
        b = tk.Button(parent, text=text, command=cmd,
                      font=(FONT, size, "bold"), bg=color, fg=C['white'],
                      relief="flat", padx=pad_x, pady=pad_y, cursor="hand2",
                      activebackground=hover, activeforeground=C['white'],
                      disabledforeground=C['text_dim'], bd=0)

        def _on_enter(e):
            if b['state'] == 'normal':
                b.config(bg=hover)
                b.after(80, lambda: b.config(bg=hover) if b.winfo_exists() else None)

        def _on_leave(e):
            if b['state'] == 'normal':
                b.config(bg=color)

        b.bind("<Enter>", _on_enter)
        b.bind("<Leave>", _on_leave)
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

    def _redraw_conf_bar(self):
        c = self.conf_bar_bg
        c.delete("all")
        w = c.winfo_width()
        h = 10
        # Track
        c.create_rectangle(0, 2, w, h - 2, fill=C['border'], outline="", tags="track")
        # Fill
        fill_w = int(w * self._conf_pct / 100)
        if fill_w > 4:
            c.create_rectangle(0, 2, fill_w, h - 2, fill=self._conf_color, outline="", tags="fill")
            # Shine highlight
            c.create_rectangle(0, 2, fill_w, 5, fill=self._conf_color, outline="",
                               stipple="gray50", tags="shine")

    def _update_conf_bar(self, pct, color):
        self._conf_pct = pct
        self._conf_color = color
        self.conf_bar_bg.update_idletasks()
        self._redraw_conf_bar()

    # ─── FILE UPLOAD ──────────────────────────────────────────────────────────

    def upload_images(self):
        path = filedialog.askopenfilename(
            title="Select a Crop Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )
        if not path:
            return
        self.image_paths = [path]
        self.csv_path, self.file_type = None, 'image'
        self._display_image_grid()
        self._enable_detect()
        self.img_btn.config(text="🔄  Replace Image", bg=C['accent'])
        self.multi_btn.config(text="🗂  Multiple Images", bg=C['primary'])
        self.add_btn.pack(side="left", padx=(0, 10))
        self.csv_btn.config(text="📊 Dataset (CSV)", bg=C['warning'])
        self.count_label.config(text=f"{len(self.image_paths)} image loaded")
        self._reset_results()
        self._set_status(f"LOADED 1 IMAGE")

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
        self.multi_btn.config(text=f"🔄  Replace Batch", bg=C['primary'])
        self.img_btn.config(text="🖼  Single Image", bg=C['accent'])
        self.add_btn.pack(side="left", padx=(0, 10))
        self.csv_btn.config(text="📊 Dataset (CSV)", bg=C['warning'])
        self.count_label.config(text=f"{len(self.image_paths)} images loaded")
        self._reset_results()
        self._set_status(f"LOADED {len(self.image_paths)} IMAGES")

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
        self.count_label.config(text=f"{len(self.image_paths)} image(s) loaded")
        self._reset_results()
        self._set_status(f"{len(self.image_paths)} IMAGE(S) LOADED  (+{len(new)} ADDED)")

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
            self.csv_btn.config(text="DATASET ACTIVE", bg=C['success'])
            self.img_btn.config(text="🖼  Single Image", bg=C['accent'])
            self.multi_btn.config(text="🗂  Multiple Images", bg=C['primary'])
            self.add_btn.pack_forget()
            self.count_label.config(text="")
            self._reset_results()
            self._set_status(f"DATASET ACTIVE: {os.path.basename(path)}")
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
        sb = tk.Scrollbar(self.display_frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas, bg=C['surface'])
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_resize(e):
            canvas.itemconfig(win_id, width=e.width)
        canvas.bind("<Configure>", _on_resize)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind_all("<MouseWheel>", lambda e, c=canvas: c.yview_scroll(-1*(e.delta//120), "units") if c.winfo_exists() else None)

        COLS = 4
        for idx, path in enumerate(self.image_paths):
            row, col = divmod(idx, COLS)
            # Outer wrapper for relative positioning of the ✕ button
            wrapper = tk.Frame(inner, bg=C['surface'])
            wrapper.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            inner.columnconfigure(col, weight=1)

            cell = tk.Frame(wrapper, bg=C['card'],
                            highlightthickness=1, highlightbackground=C['border'])
            cell.pack(fill="both", expand=True, padx=0, pady=0)

            try:
                img = Image.open(path)
                img.thumbnail((160, 120), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self._thumb_refs.append(photo)
                tk.Label(cell, image=photo, bg=C['card']).pack(pady=(22, 4), padx=10)
            except Exception:
                tk.Label(cell, text="CORRUPT DATA", bg=C['card'], fg=C['danger'],
                         font=(FONT, 8, "bold")).pack(pady=(22, 4))

            tk.Label(cell, text=os.path.basename(path), font=(FONT, 7),
                     bg=C['card'], fg=C['text_dim'], wraplength=140).pack(pady=(0, 10))

            # ✕ close button — top-right overlay
            close_btn = tk.Button(
                cell, text="✕", font=(FONT, 9, "bold"),
                bg=C['danger_dim'], fg=C['danger'], relief="flat",
                padx=5, pady=1, cursor="hand2", bd=0,
                activebackground=C['danger'], activeforeground=C['white'],
                command=lambda p=path: self._remove_image(p)
            )
            close_btn.place(relx=1.0, rely=0.0, anchor="ne", x=-6, y=6)
            close_btn.bind("<Enter>", lambda e, b=close_btn: b.config(bg=C['danger'], fg=C['white']))
            close_btn.bind("<Leave>", lambda e, b=close_btn: b.config(bg=C['danger_dim'], fg=C['danger']))

    def _remove_image(self, path):
        self.image_paths.remove(path)
        if not self.image_paths:
            self._clear_display()
            self.display_label = tk.Label(
                self.display_frame,
                text="SYSTEM IDLE\n\nNo samples detected in active buffer.\nPlease provide agricultural imagery to resume.",
                font=(FONT, 10), bg=C['surface'], fg=C['text_dim'], justify="center")
            self.display_label.pack(expand=True)
            self.detect_btn.config(state="disabled", bg=C['text_dim'])
            self.img_btn.config(text="🖼  Single Image", bg=C['accent'])
            self.multi_btn.config(text="🗂  Multiple Images", bg=C['primary'])
            self.add_btn.pack_forget()
            self.count_label.config(text="")
            self._reset_results()
            self._set_status("READY")
        else:
            self.img_btn.config(text="🔄  Replace Images", bg=C['accent'])
            self._set_status(f"BUFFER: {len(self.image_paths)} IMAGE(S) REMAINING")
            self.count_label.config(text=f"{len(self.image_paths)} image(s) loaded")
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

        tk.Label(f, text="📊", font=(FONT, 32), bg=C['surface']).pack(pady=(20, 8))
        tk.Label(f, text=os.path.basename(path).upper(),
                 font=(FONT, 11, "bold"), bg=C['surface'], fg=C['text']).pack()
        tk.Label(f, text=f"METRIC: {rows:,} RECORDS | {cols} FIELDS",
                 font=(FONT, 8, "bold"), bg=C['surface'], fg=C['text_dim']).pack(pady=10)

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
        has_name = any("name" in r for r in results)
        hdr = tk.Frame(table_wrap, bg=C['surface'])
        hdr.pack(fill="x", pady=(0, 4))
        cols = [("#", 5), ("NAME", 20), ("DIAGNOSIS", 18), ("CONFIDENCE", 13), ("STATUS", 8)] if has_name \
            else [("INDEX", 8), ("DIAGNOSIS", 22), ("CONFIDENCE", 15), ("STATUS", 10)]
        for text, w in cols:
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

            if has_name:
                tk.Label(row_frame, text=idx_str, font=(FONT, 9, "bold"), bg=row_bg,
                         fg=C['text_dim'], width=5, anchor="w", padx=10, pady=6).pack(side="left")
                tk.Label(row_frame, text=r.get("name", ""), font=(FONT, 9), bg=row_bg,
                         fg=C['text'], width=20, anchor="w", padx=10).pack(side="left")
                tk.Label(row_frame, text=r["disease"].upper(), font=(FONT, 9, "bold"), bg=row_bg,
                         fg=color, width=18, anchor="w", padx=10).pack(side="left")
                tk.Label(row_frame, text=conf_str, font=(FONT, 9), bg=row_bg,
                         fg=C['text'], width=13, anchor="w", padx=10).pack(side="left")
                tk.Label(row_frame, text=status_icon, font=(FONT, 10), bg=row_bg,
                         fg=color, width=8, anchor="w", padx=10).pack(side="left")
            else:
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
            err_msg = str(e)
            self.root.after(0, lambda m=err_msg: self._analysis_error(m))

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
        canvas.bind_all("<MouseWheel>", lambda e, c=canvas: c.yview_scroll(-1*(e.delta//120), "units") if c.winfo_exists() else None)

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
        self.conf_badge.config(text=f"{conf}%", bg=C['card_inner'], fg=color)
        self.root.after(50, lambda: self._update_conf_bar(conf, color))
        self._set_remedy(status_text)

    def _reset_results(self):
        self.disease_label.config(text="—", fg=C['text_muted'])
        self.conf_badge.config(text="0%", bg=C['card_inner'], fg=C['text_dim'])
        self._conf_pct = 0
        self._conf_color = C['primary']
        self._redraw_conf_bar()
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
