import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import pandas as pd
import os
import threading
from disease_detector import DiseaseDetector

C = {
    'bg':           '#0f172a',
    'surface':      '#1e293b',
    'card':         '#263348',
    'border':       '#334155',
    'primary':      '#3b82f6',
    'primary_h':    '#2563eb',
    'success':      '#22c55e',
    'success_h':    '#16a34a',
    'warning':      '#f59e0b',
    'warning_h':    '#d97706',
    'danger':       '#ef4444',
    'accent':       '#06b6d4',
    'accent_h':     '#0891b2',
    'text':         '#f1f5f9',
    'text_muted':   '#94a3b8',
    'text_dim':     '#475569',
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

        self.image_path = None
        self.csv_path = None
        self.csv_data = None
        self.file_type = None
        self.detector = DiseaseDetector()

        self._build_ui()

    # ─── UI CONSTRUCTION ──────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_header()
        self._build_body()
        self._build_statusbar()

    def _build_header(self):
        hdr = tk.Frame(self.root, bg=C['surface'], height=72)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        inner = tk.Frame(hdr, bg=C['surface'])
        inner.pack(expand=True, fill="both", padx=32)

        tk.Label(inner, text="🌿  Crop Disease Detection",
                 font=(FONT, 22, "bold"), bg=C['surface'], fg=C['text']).pack(side="left", pady=16)

        badge = tk.Label(inner, text="  AI Powered  ",
                         font=(FONT, 9, "bold"), bg=C['primary'], fg=C['white'],
                         padx=8, pady=3)
        badge.pack(side="left", padx=12, pady=20)

        tk.Label(inner, text="Plant Health Analysis System",
                 font=(FONT, 10), bg=C['surface'], fg=C['text_muted']).pack(side="right", pady=20)

        sep = tk.Frame(self.root, bg=C['primary'], height=2)
        sep.pack(fill="x")

    def _build_body(self):
        body = tk.Frame(self.root, bg=C['bg'])
        body.pack(expand=True, fill="both", padx=24, pady=20)

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

        self.img_btn = self._btn(btn_row, "📷  Upload Image", C['accent'], C['accent_h'], self.upload_image)
        self.img_btn.pack(side="left", padx=(0, 10))

        self.csv_btn = self._btn(btn_row, "📊  Upload CSV", C['warning'], C['warning_h'], self.upload_csv)
        self.csv_btn.pack(side="left")

        # Preview card
        prev_card = self._card(left, "🖼️  Preview")
        prev_card.pack(fill="both", expand=True)

        self.display_frame = tk.Frame(prev_card, bg=C['surface'],
                                      highlightthickness=1, highlightbackground=C['border'])
        self.display_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        self.display_label = tk.Label(self.display_frame,
                                      text="📷\n\nNo file selected\nUpload an image or CSV to begin",
                                      font=(FONT, 11), bg=C['surface'], fg=C['text_dim'],
                                      justify="center")
        self.display_label.pack(expand=True)

    def _build_right(self, parent):
        right = tk.Frame(parent, bg=C['bg'], width=400)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        # Analyse card
        ctrl_card = self._card(right, "🔬  Analysis")
        ctrl_card.pack(fill="x", pady=(0, 14))

        self.detect_btn = self._btn(ctrl_card, "🔍  Start Analysis", C['success'], C['success_h'],
                                    self.detect_disease, big=True)
        self.detect_btn.pack(fill="x", padx=20, pady=(0, 18))
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
        bar = tk.Frame(self.root, bg=C['surface'], height=28)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(bar, textvariable=self.status_var,
                 font=(FONT, 8), bg=C['surface'], fg=C['text_muted']).pack(side="left", padx=16)

        tk.Label(bar, text="Crop Disease Detection System  v1.0",
                 font=(FONT, 8), bg=C['surface'], fg=C['text_dim']).pack(side="right", padx=16)

    # ─── HELPERS ──────────────────────────────────────────────────────────────

    def _card(self, parent, title):
        frame = tk.Frame(parent, bg=C['card'],
                         highlightthickness=1, highlightbackground=C['border'])
        tk.Label(frame, text=title, font=(FONT, 11, "bold"),
                 bg=C['card'], fg=C['text']).pack(anchor="w", padx=20, pady=(16, 12))
        sep = tk.Frame(frame, bg=C['border'], height=1)
        sep.pack(fill="x", padx=20, pady=(0, 14))
        return frame

    def _btn(self, parent, text, color, hover, cmd, big=False):
        size = 12 if big else 10
        pad_y = 13 if big else 10
        b = tk.Button(parent, text=text, command=cmd,
                      font=(FONT, size, "bold"), bg=color, fg=C['white'],
                      relief="flat", padx=20, pady=pad_y, cursor="hand2",
                      activebackground=hover, activeforeground=C['white'],
                      disabledforeground="#6b7280")
        b.bind("<Enter>", lambda e: b.config(bg=hover) if b['state'] == 'normal' else None)
        b.bind("<Leave>", lambda e: b.config(bg=color) if b['state'] == 'normal' else None)
        b._color = color
        return b

    def _label(self, parent, text, color, size):
        return tk.Label(parent, text=text, font=(FONT, size, "bold"),
                        bg=C['card'], fg=color)

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
        path = filedialog.askopenfilename(
            title="Select Crop Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )
        if not path:
            return
        self.image_path, self.csv_path, self.file_type = path, None, 'image'
        self._display_image(path)
        self._enable_detect()
        self.img_btn.config(text="✅  Image Loaded", bg=C['success'])
        self.csv_btn.config(text="📊  Upload CSV", bg=C['warning'])
        self._reset_results()
        self._set_status(f"Loaded: {os.path.basename(path)}")

    def upload_csv(self):
        path = filedialog.askopenfilename(
            title="Select CSV / Excel File",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            self.csv_data = pd.read_excel(path) if path.endswith(('.xlsx', '.xls')) else pd.read_csv(path)
            self.csv_path, self.image_path, self.file_type = path, None, 'csv'
            self._display_csv(path)
            self._enable_detect()
            self.csv_btn.config(text="✅  CSV Loaded", bg=C['success'])
            self.img_btn.config(text="📷  Upload Image", bg=C['accent'])
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

    def _display_image(self, path, label_text=None):
        try:
            img = Image.open(path)
            img.thumbnail((480, 340), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self._clear_display()
            lbl = tk.Label(self.display_frame, image=photo, bg=C['surface'])
            lbl.pack(expand=True, pady=(12, 4))
            lbl.image = photo
            tk.Label(self.display_frame,
                     text=label_text or f"📷  {os.path.basename(path)}",
                     font=(FONT, 9), bg=C['surface'], fg=C['text_muted']).pack(pady=(0, 10))
        except Exception as e:
            messagebox.showerror("Error", f"Cannot display image:\n{e}")

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
        # Header
        hdr = tk.Frame(self.display_frame, bg=C['surface'])
        hdr.pack(fill="x", padx=8, pady=(10, 4))
        for text, w in [("Row", 5), ("Disease", 20), ("Confidence", 12), ("Status", 8)]:
            tk.Label(hdr, text=text, font=(FONT, 9, "bold"), bg=C['border'], fg=C['text'],
                     width=w, anchor="w", padx=6, pady=4).pack(side="left", padx=1)

        # Scrollable rows
        container = tk.Frame(self.display_frame, bg=C['surface'])
        container.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        canvas = tk.Canvas(container, bg=C['surface'], highlightthickness=0)
        sb = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=C['surface'])
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
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
            row_bg = C['card'] if i % 2 == 0 else C['surface']
            color = disease_colors.get(r["disease"], C['text_muted'])
            row_frame = tk.Frame(inner, bg=row_bg)
            row_frame.pack(fill="x", pady=1)
            for text, w in [(str(i + 1), 5), (r["disease"], 20),
                            (f"{r['confidence']}%", 12),
                            ("✅" if r["disease"] == "Healthy" else "⚠️", 8)]:
                tk.Label(row_frame, text=text, font=(FONT, 9), bg=row_bg,
                         fg=color if text not in (str(i + 1),) else C['text_muted'],
                         width=w, anchor="w", padx=6, pady=3).pack(side="left", padx=1)

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
                disease, conf, remedy, marked = self.detector.analyze_image(self.image_path)
                self.root.after(0, lambda: self._finish_image(disease, conf, remedy, marked))
            else:
                disease, conf, remedy, rows = self.detector.analyze_csv(self.csv_data)
                self.root.after(0, lambda: self._finish_csv(disease, conf, remedy, rows))
        except Exception as e:
            self.root.after(0, lambda: self._analysis_error(str(e)))

    def _finish_image(self, disease, conf, remedy, marked):
        self._display_image_obj(marked, f"🔍  Analysis complete — {os.path.basename(self.image_path)}")
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
            label = "No Disease Detected"
            status_text = f"✅ HEALTHY  —  Confidence: {conf}%\n\n"
            status_text += "RECOMMENDATION:\n" + "─" * 32 + "\n" + remedy
        else:
            color = C['danger'] if conf >= 80 else C['warning']
            icon = "⚠️"
            label = disease
            status_text = f"⚠️ DISEASE DETECTED  —  {disease}\nConfidence: {conf}%\n\n"
            status_text += "TREATMENT:\n" + "─" * 32 + "\n" + remedy

        self.disease_label.config(text=f"{icon}  {label}", fg=color)
        self.conf_badge.config(text=f"{conf}%", bg=color)
        self.root.after(50, lambda: self._update_conf_bar(conf, color))
        self._set_remedy(status_text)

    def _reset_results(self):
        self.disease_label.config(text="—", fg=C['text_muted'])
        self.conf_badge.config(text="0%", bg=C['text_dim'])
        self.conf_bar.place(x=0, y=0, relheight=1, width=0)
        self._set_remedy("Upload a file and run analysis to receive treatment recommendations.")


def main():
    root = tk.Tk()
    w, h = 1280, 820
    x = (root.winfo_screenwidth() - w) // 2
    y = (root.winfo_screenheight() - h) // 2
    root.geometry(f"{w}x{h}+{x}+{y}")

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("Horizontal.TProgressbar",
                    troughcolor=C['border'], background=C['primary'],
                    thickness=6, borderwidth=0)

    CropDiseaseDetector(root)
    root.mainloop()

if __name__ == "__main__":
    main()
