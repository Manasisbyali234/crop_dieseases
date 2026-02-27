import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import pandas as pd
import os
from disease_detector import DiseaseDetector

class CropDiseaseDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("🌱 Crop Disease Detection System")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f8f9fa")
        self.root.resizable(True, True)
        self.root.minsize(1000, 700)
        
        # Variables
        self.image_path = None
        self.csv_path = None
        self.csv_data = None
        self.file_type = None
        
        # Initialize disease detector
        self.detector = DiseaseDetector()
        
        # Professional color scheme
        self.colors = {
            'primary': '#2563eb',
            'primary_dark': '#1e40af',
            'secondary': '#64748b',
            'accent': '#06b6d4',
            'accent_hover': '#0891b2',
            'success': '#10b981',
            'success_hover': '#059669',
            'danger': '#ef4444',
            'warning': '#f59e0b',
            'warning_hover': '#d97706',
            'light': '#f8fafc',
            'white': '#ffffff',
            'dark': '#0f172a',
            'muted': '#94a3b8',
            'card': '#ffffff',
            'border': '#e2e8f0',
            'shadow': '#00000015'
        }
        
        self.setup_ui()
    
    def setup_ui(self):
        # Header with gradient effect
        header_frame = tk.Frame(self.root, bg=self.colors['primary'], height=90)
        header_frame.pack(fill="x")
        header_frame.pack_propagate(False)
        
        title = tk.Label(header_frame, 
                        text="🌱 Crop Disease Detection System",
                        font=("Segoe UI", 26, "bold"),
                        bg=self.colors['primary'],
                        fg=self.colors['white'])
        title.pack(expand=True, pady=5)
        
        subtitle = tk.Label(header_frame,
                           text="AI-Powered Plant Health Analysis",
                           font=("Segoe UI", 10),
                           bg=self.colors['primary'],
                           fg="#93c5fd")
        subtitle.pack()
        
        # Main container
        main_container = tk.Frame(self.root, bg=self.colors['light'])
        main_container.pack(expand=True, fill="both", padx=25, pady=25)
        
        # Left panel with shadow effect
        left_panel = tk.Frame(main_container, bg=self.colors['card'], relief="flat", bd=0, highlightthickness=1, highlightbackground=self.colors['border'])
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 12))
        
        # Upload section
        upload_frame = tk.Frame(left_panel, bg=self.colors['card'])
        upload_frame.pack(fill="x", padx=25, pady=25)
        
        tk.Label(upload_frame, text="📁 File Upload", 
                font=("Segoe UI", 15, "bold"),
                bg=self.colors['card'],
                fg=self.colors['dark']).pack(anchor="w", pady=(0, 15))
        
        btn_frame = tk.Frame(upload_frame, bg=self.colors['card'])
        btn_frame.pack(fill="x")
        
        self.upload_img_btn = tk.Button(btn_frame, 
                                       text="📷 Upload Image",
                                       command=self.upload_image,
                                       font=("Segoe UI", 10, "bold"),
                                       bg=self.colors['accent'],
                                       fg=self.colors['white'],
                                       relief="flat",
                                       padx=25, pady=12,
                                       cursor="hand2",
                                       activebackground=self.colors['accent_hover'],
                                       activeforeground=self.colors['white'])
        self.upload_img_btn.pack(side="left", padx=(0, 12))
        self.upload_img_btn.bind("<Enter>", lambda e: self.upload_img_btn.config(bg=self.colors['accent_hover']))
        self.upload_img_btn.bind("<Leave>", lambda e: self.upload_img_btn.config(bg=self.colors['accent']))
        
        self.upload_csv_btn = tk.Button(btn_frame,
                                       text="📊 Upload CSV",
                                       command=self.upload_csv,
                                       font=("Segoe UI", 10, "bold"),
                                       bg=self.colors['warning'],
                                       fg=self.colors['white'],
                                       relief="flat",
                                       padx=25, pady=12,
                                       cursor="hand2",
                                       activebackground=self.colors['warning_hover'],
                                       activeforeground=self.colors['white'])
        self.upload_csv_btn.pack(side="left")
        self.upload_csv_btn.bind("<Enter>", lambda e: self.upload_csv_btn.config(bg=self.colors['warning_hover']))
        self.upload_csv_btn.bind("<Leave>", lambda e: self.upload_csv_btn.config(bg=self.colors['warning']))
        
        # Preview section
        preview_frame = tk.Frame(left_panel, bg=self.colors['card'])
        preview_frame.pack(fill="both", expand=True, padx=25, pady=(15, 25))
        
        tk.Label(preview_frame, text="🖼️ Preview", 
                font=("Segoe UI", 15, "bold"),
                bg=self.colors['card'],
                fg=self.colors['dark']).pack(anchor="w", pady=(0, 15))
        
        # Display area with rounded effect
        self.display_frame = tk.Frame(preview_frame, bg=self.colors['light'],
                                     relief="flat", bd=0, highlightthickness=2, highlightbackground=self.colors['border'])
        self.display_frame.pack(fill="both", expand=True)
        
        self.display_label = tk.Label(self.display_frame,
                                     text="📷 📊\n\nNo file selected\n\nUpload an image or CSV file to begin",
                                     bg=self.colors['light'],
                                     font=("Segoe UI", 11),
                                     fg=self.colors['muted'],
                                     justify="center")
        self.display_label.pack(expand=True)
        
        # Right panel with shadow
        right_panel = tk.Frame(main_container, bg=self.colors['card'],
                              relief="flat", bd=0, width=420, highlightthickness=1, highlightbackground=self.colors['border'])
        right_panel.pack(side="right", fill="y")
        right_panel.pack_propagate(False)
        
        # Analysis section
        analysis_frame = tk.Frame(right_panel, bg=self.colors['card'])
        analysis_frame.pack(fill="x", padx=25, pady=25)
        
        tk.Label(analysis_frame, text="🔬 Disease Analysis", 
                font=("Segoe UI", 15, "bold"),
                bg=self.colors['card'],
                fg=self.colors['dark']).pack(anchor="w", pady=(0, 18))
        
        self.detect_btn = tk.Button(analysis_frame,
                                   text="🔍 Start Analysis",
                                   command=self.detect_disease,
                                   font=("Segoe UI", 12, "bold"),
                                   bg=self.colors['success'],
                                   fg=self.colors['white'],
                                   relief="flat",
                                   padx=35, pady=14,
                                   cursor="hand2",
                                   state="disabled",
                                   disabledforeground="#d1d5db",
                                   activebackground=self.colors['success_hover'],
                                   activeforeground=self.colors['white'])
        self.detect_btn.pack(fill="x")
        self.detect_btn.bind("<Enter>", lambda e: self.detect_btn.config(bg=self.colors['success_hover']) if self.detect_btn['state'] == 'normal' else None)
        self.detect_btn.bind("<Leave>", lambda e: self.detect_btn.config(bg=self.colors['success']) if self.detect_btn['state'] == 'normal' else None)
        
        # Results section
        results_frame = tk.Frame(right_panel, bg=self.colors['card'])
        results_frame.pack(fill="both", expand=True, padx=25, pady=(15, 25))
        
        tk.Label(results_frame, text="📊 Results", 
                font=("Segoe UI", 15, "bold"),
                bg=self.colors['card'],
                fg=self.colors['dark']).pack(anchor="w", pady=(0, 18))
        
        # Disease result card
        disease_frame = tk.Frame(results_frame, bg=self.colors['light'], relief="flat", bd=0, highlightthickness=1, highlightbackground=self.colors['border'])
        disease_frame.pack(fill="x", pady=(0, 12))
        
        tk.Label(disease_frame, text="🦠 Disease", 
                font=("Segoe UI", 9, "bold"),
                bg=self.colors['light'],
                fg=self.colors['secondary']).pack(anchor="w", padx=15, pady=(10, 2))
        
        self.disease_label = tk.Label(disease_frame, text="Awaiting analysis...",
                                     font=("Segoe UI", 13, "bold"),
                                     bg=self.colors['light'],
                                     fg=self.colors['muted'])
        self.disease_label.pack(anchor="w", padx=15, pady=(0, 10))
        
        # Confidence result card
        confidence_frame = tk.Frame(results_frame, bg=self.colors['light'], relief="flat", bd=0, highlightthickness=1, highlightbackground=self.colors['border'])
        confidence_frame.pack(fill="x", pady=(0, 12))
        
        tk.Label(confidence_frame, text="📈 Confidence Level", 
                font=("Segoe UI", 9, "bold"),
                bg=self.colors['light'],
                fg=self.colors['secondary']).pack(anchor="w", padx=15, pady=(10, 2))
        
        self.confidence_label = tk.Label(confidence_frame, text="0%",
                                        font=("Segoe UI", 13, "bold"),
                                        bg=self.colors['light'],
                                        fg=self.colors['muted'])
        self.confidence_label.pack(anchor="w", padx=15, pady=(0, 10))
        
        # Treatment recommendations card
        remedy_frame = tk.Frame(results_frame, bg=self.colors['light'], relief="flat", bd=0, highlightthickness=1, highlightbackground=self.colors['border'])
        remedy_frame.pack(fill="both", expand=True)
        
        tk.Label(remedy_frame, text="💊 Treatment Recommendations", 
                font=("Segoe UI", 9, "bold"),
                bg=self.colors['light'],
                fg=self.colors['secondary']).pack(anchor="w", padx=15, pady=(10, 8))
        
        text_container = tk.Frame(remedy_frame, bg=self.colors['light'])
        text_container.pack(fill="both", expand=True, padx=15, pady=(0, 10))
        
        self.remedy_text = tk.Text(text_container,
                                  height=8,
                                  wrap="word",
                                  font=("Segoe UI", 9),
                                  bg=self.colors['white'],
                                  relief="flat",
                                  bd=0,
                                  padx=12,
                                  pady=10,
                                  highlightthickness=1,
                                  highlightbackground=self.colors['border'])
        
        scrollbar = tk.Scrollbar(text_container, orient="vertical", command=self.remedy_text.yview)
        self.remedy_text.configure(yscrollcommand=scrollbar.set)
        
        self.remedy_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.remedy_text.insert(1.0, "Upload a file and run analysis to receive treatment recommendations.")
        self.remedy_text.config(state="disabled")
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Crop Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.image_path = file_path
            self.csv_path = None
            self.file_type = 'image'
            self.display_image()
            self.detect_btn.config(state="normal", bg=self.colors['success'])
            self.upload_img_btn.config(text="✅ Image Loaded", bg=self.colors['success'])
            self.upload_csv_btn.config(text="📊 Upload CSV", bg=self.colors['warning'])
            self.reset_results()
            messagebox.showinfo("Success", f"Image loaded: {os.path.basename(file_path)}")
    
    def upload_csv(self):
        file_path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx *.xls"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                if file_path.endswith(('.xlsx', '.xls')):
                    self.csv_data = pd.read_excel(file_path)
                else:
                    self.csv_data = pd.read_csv(file_path)
                
                self.csv_path = file_path
                self.image_path = None
                self.file_type = 'csv'
                self.display_csv()
                self.detect_btn.config(state="normal", bg=self.colors['success'])
                self.upload_csv_btn.config(text="✅ CSV Loaded", bg=self.colors['success'])
                self.upload_img_btn.config(text="📷 Upload Image", bg=self.colors['accent'])
                self.reset_results()
                
                rows, cols = self.csv_data.shape
                messagebox.showinfo("Success", f"CSV loaded: {os.path.basename(file_path)}\n{rows} rows, {cols} columns")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    
    def display_image(self):
        try:
            image = Image.open(self.image_path)
            image.thumbnail((400, 300), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            
            for widget in self.display_frame.winfo_children():
                widget.destroy()
            
            img_label = tk.Label(self.display_frame, image=photo, bg=self.colors['light'])
            img_label.pack(expand=True)
            img_label.image = photo
            
            filename = os.path.basename(self.image_path)
            info_label = tk.Label(self.display_frame, text=f"📷 {filename}",
                                 font=("Segoe UI", 9),
                                 bg=self.colors['light'],
                                 fg=self.colors['secondary'])
            info_label.pack(pady=8)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display image: {str(e)}")
    
    def display_csv(self):
        try:
            for widget in self.display_frame.winfo_children():
                widget.destroy()
            
            filename = os.path.basename(self.csv_path)
            rows, cols = self.csv_data.shape
            
            info_frame = tk.Frame(self.display_frame, bg=self.colors['light'])
            info_frame.pack(expand=True)
            
            tk.Label(info_frame, text=f"📊 {filename}",
                    font=("Segoe UI", 13, "bold"),
                    bg=self.colors['light'],
                    fg=self.colors['dark']).pack(pady=12)
            
            tk.Label(info_frame, text=f"📈 {rows:,} rows, {cols} columns",
                    font=("Segoe UI", 10),
                    bg=self.colors['light'],
                    fg=self.colors['secondary']).pack(pady=5)
            
            # Show first few columns
            columns_text = "Columns: " + ", ".join(self.csv_data.columns[:5].tolist())
            if len(self.csv_data.columns) > 5:
                columns_text += f"... (+{len(self.csv_data.columns)-5} more)"
            
            tk.Label(info_frame, text=columns_text,
                    font=("Segoe UI", 9),
                    bg=self.colors['light'],
                    fg=self.colors['muted'],
                    wraplength=350).pack(pady=12)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display CSV: {str(e)}")
    
    def detect_disease(self):
        if not self.image_path and not self.csv_path:
            messagebox.showwarning("Warning", "Please upload a file first!")
            return
        
        if self.file_type == 'image':
            self.analyze_image()
        elif self.file_type == 'csv':
            self.analyze_csv()
    
    def analyze_image(self):
        try:
            disease_name, confidence, remedy, marked_image = self.detector.analyze_image(self.image_path)
            
            # Display marked image
            marked_image.thumbnail((400, 300), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(marked_image)
            
            for widget in self.display_frame.winfo_children():
                widget.destroy()
            
            img_label = tk.Label(self.display_frame, image=photo, bg=self.colors['light'])
            img_label.pack(expand=True)
            img_label.image = photo
            
            self.display_results(disease_name, confidence, remedy)
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
    
    def analyze_csv(self):
        try:
            disease_name, confidence, remedy = self.detector.analyze_csv(self.csv_data)
            self.display_results(disease_name, confidence, remedy)
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
    
    def display_results(self, disease_name, confidence, remedy):
        # Check if crop is healthy
        if disease_name == "Healthy":
            self.disease_label.config(text="✅ No Disease Detected", font=("Segoe UI", 13, "bold"), fg=self.colors['success'])
            self.confidence_label.config(text=f"{confidence}%", font=("Segoe UI", 13, "bold"), fg=self.colors['success'])
            
            # Update remedy text for healthy crop
            self.remedy_text.config(state="normal")
            self.remedy_text.delete(1.0, tk.END)
            
            formatted_remedy = f"✅ ANALYSIS COMPLETE\n\n"
            formatted_remedy += f"Status: Healthy Crop\n"
            formatted_remedy += f"Confidence: {confidence}%\n\n"
            formatted_remedy += "RECOMMENDATION:\n"
            formatted_remedy += "─" * 30 + "\n"
            formatted_remedy += remedy
            
            self.remedy_text.insert(1.0, formatted_remedy)
            self.remedy_text.config(state="disabled")
            
            messagebox.showinfo("Analysis Complete", 
                               f"✅ No Disease Detected\nConfidence: {confidence}%\n\nYour crop appears healthy!")
        else:
            # Disease detected
            self.disease_label.config(text=f"⚠️ {disease_name}", font=("Segoe UI", 13, "bold"))
            self.confidence_label.config(text=f"{confidence}%", font=("Segoe UI", 13, "bold"))
            
            # Set colors based on confidence
            if confidence >= 80:
                color = self.colors['danger']
            else:
                color = self.colors['warning']
            
            self.disease_label.config(fg=color)
            self.confidence_label.config(fg=color)
            
            # Update remedy text
            self.remedy_text.config(state="normal")
            self.remedy_text.delete(1.0, tk.END)
            
            formatted_remedy = f"⚠️ DISEASE DETECTED\n\n"
            formatted_remedy += f"Disease: {disease_name}\n"
            formatted_remedy += f"Confidence: {confidence}%\n\n"
            formatted_remedy += "TREATMENT:\n"
            formatted_remedy += "─" * 30 + "\n"
            formatted_remedy += remedy
            
            self.remedy_text.insert(1.0, formatted_remedy)
            self.remedy_text.config(state="disabled")
            
            messagebox.showinfo("Disease Detected", 
                               f"⚠️ Disease: {disease_name}\nConfidence: {confidence}%\n\nCheck results panel for treatment details.")
    
    def reset_results(self):
        self.disease_label.config(text="Awaiting analysis...", fg=self.colors['muted'])
        self.confidence_label.config(text="0%", fg=self.colors['muted'])
        
        self.remedy_text.config(state="normal")
        self.remedy_text.delete(1.0, tk.END)
        self.remedy_text.insert(1.0, "Upload a file and run analysis to receive treatment recommendations.")
        self.remedy_text.config(state="disabled")

def main():
    root = tk.Tk()
    
    # Center window
    root.update_idletasks()
    width = 1200
    height = 800
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")
    
    app = CropDiseaseDetector(root)
    root.mainloop()

if __name__ == "__main__":
    main()