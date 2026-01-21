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
            'primary': '#1e3a8a',
            'secondary': '#64748b',
            'accent': '#0ea5e9',
            'success': '#059669',
            'danger': '#dc2626',
            'warning': '#d97706',
            'light': '#f1f5f9',
            'white': '#ffffff',
            'dark': '#0f172a',
            'muted': '#94a3b8',
            'card': '#ffffff',
            'border': '#e2e8f0'
        }
        
        self.setup_ui()
    
    def setup_ui(self):
        # Header
        header_frame = tk.Frame(self.root, bg=self.colors['primary'], height=80)
        header_frame.pack(fill="x")
        header_frame.pack_propagate(False)
        
        title = tk.Label(header_frame, 
                        text="🌱 Crop Disease Detection System",
                        font=("Arial", 24, "bold"),
                        bg=self.colors['primary'],
                        fg=self.colors['white'])
        title.pack(expand=True)
        
        # Main container
        main_container = tk.Frame(self.root, bg=self.colors['light'])
        main_container.pack(expand=True, fill="both", padx=20, pady=20)
        
        # Left panel
        left_panel = tk.Frame(main_container, bg=self.colors['card'], relief="solid", bd=1)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        # Upload section
        upload_frame = tk.Frame(left_panel, bg=self.colors['card'])
        upload_frame.pack(fill="x", padx=20, pady=20)
        
        tk.Label(upload_frame, text="📁 File Upload", 
                font=("Arial", 14, "bold"),
                bg=self.colors['card'],
                fg=self.colors['primary']).pack(anchor="w", pady=(0, 10))
        
        btn_frame = tk.Frame(upload_frame, bg=self.colors['card'])
        btn_frame.pack(fill="x")
        
        self.upload_img_btn = tk.Button(btn_frame, 
                                       text="📷 Upload Image",
                                       command=self.upload_image,
                                       font=("Arial", 10, "bold"),
                                       bg=self.colors['accent'],
                                       fg=self.colors['white'],
                                       relief="flat",
                                       padx=20, pady=10,
                                       cursor="hand2")
        self.upload_img_btn.pack(side="left", padx=(0, 10))
        
        self.upload_csv_btn = tk.Button(btn_frame,
                                       text="📊 Upload CSV",
                                       command=self.upload_csv,
                                       font=("Arial", 10, "bold"),
                                       bg=self.colors['warning'],
                                       fg=self.colors['white'],
                                       relief="flat",
                                       padx=20, pady=10,
                                       cursor="hand2")
        self.upload_csv_btn.pack(side="left")
        
        # Preview section
        preview_frame = tk.Frame(left_panel, bg=self.colors['card'])
        preview_frame.pack(fill="both", expand=True, padx=20, pady=(10, 20))
        
        tk.Label(preview_frame, text="🖼️ Preview", 
                font=("Arial", 14, "bold"),
                bg=self.colors['card'],
                fg=self.colors['primary']).pack(anchor="w", pady=(0, 10))
        
        # Display area
        self.display_frame = tk.Frame(preview_frame, bg=self.colors['light'],
                                     relief="solid", bd=1)
        self.display_frame.pack(fill="both", expand=True)
        
        self.display_label = tk.Label(self.display_frame,
                                     text="📷 📊\n\nNo file selected\n\nUpload an image or CSV file to begin",
                                     bg=self.colors['light'],
                                     font=("Arial", 12),
                                     fg=self.colors['muted'],
                                     justify="center")
        self.display_label.pack(expand=True)
        
        # Right panel
        right_panel = tk.Frame(main_container, bg=self.colors['card'],
                              relief="solid", bd=1, width=400)
        right_panel.pack(side="right", fill="y")
        right_panel.pack_propagate(False)
        
        # Analysis section
        analysis_frame = tk.Frame(right_panel, bg=self.colors['card'])
        analysis_frame.pack(fill="x", padx=20, pady=20)
        
        tk.Label(analysis_frame, text="🔬 Disease Analysis", 
                font=("Arial", 14, "bold"),
                bg=self.colors['card'],
                fg=self.colors['primary']).pack(anchor="w", pady=(0, 15))
        
        self.detect_btn = tk.Button(analysis_frame,
                                   text="🔍 Start Analysis",
                                   command=self.detect_disease,
                                   font=("Arial", 12, "bold"),
                                   bg=self.colors['success'],
                                   fg=self.colors['white'],
                                   relief="flat",
                                   padx=30, pady=12,
                                   cursor="hand2",
                                   state="disabled")
        self.detect_btn.pack(fill="x")
        
        # Results section
        results_frame = tk.Frame(right_panel, bg=self.colors['card'])
        results_frame.pack(fill="both", expand=True, padx=20, pady=(10, 20))
        
        tk.Label(results_frame, text="📊 Results", 
                font=("Arial", 14, "bold"),
                bg=self.colors['card'],
                fg=self.colors['primary']).pack(anchor="w", pady=(0, 15))
        
        # Disease result
        disease_frame = tk.Frame(results_frame, bg=self.colors['light'], relief="solid", bd=1)
        disease_frame.pack(fill="x", pady=(0, 10))
        
        tk.Label(disease_frame, text="🦠 Disease:", 
                font=("Arial", 10, "bold"),
                bg=self.colors['light']).pack(anchor="w", padx=10, pady=(5, 0))
        
        self.disease_label = tk.Label(disease_frame, text="Awaiting analysis...",
                                     font=("Arial", 12, "bold"),
                                     bg=self.colors['light'],
                                     fg=self.colors['muted'])
        self.disease_label.pack(anchor="w", padx=10, pady=(0, 5))
        
        # Confidence result
        confidence_frame = tk.Frame(results_frame, bg=self.colors['light'], relief="solid", bd=1)
        confidence_frame.pack(fill="x", pady=(0, 10))
        
        tk.Label(confidence_frame, text="📈 Confidence:", 
                font=("Arial", 10, "bold"),
                bg=self.colors['light']).pack(anchor="w", padx=10, pady=(5, 0))
        
        self.confidence_label = tk.Label(confidence_frame, text="0%",
                                        font=("Arial", 12, "bold"),
                                        bg=self.colors['light'],
                                        fg=self.colors['muted'])
        self.confidence_label.pack(anchor="w", padx=10, pady=(0, 5))
        
        # Treatment recommendations
        remedy_frame = tk.Frame(results_frame, bg=self.colors['light'], relief="solid", bd=1)
        remedy_frame.pack(fill="both", expand=True)
        
        tk.Label(remedy_frame, text="💊 Treatment:", 
                font=("Arial", 10, "bold"),
                bg=self.colors['light']).pack(anchor="w", padx=10, pady=(5, 5))
        
        self.remedy_text = tk.Text(remedy_frame,
                                  height=8,
                                  wrap="word",
                                  font=("Arial", 9),
                                  bg=self.colors['light'],
                                  relief="flat",
                                  bd=0,
                                  padx=10,
                                  pady=5)
        
        scrollbar = tk.Scrollbar(remedy_frame, orient="vertical", command=self.remedy_text.yview)
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
                                 font=("Arial", 9),
                                 bg=self.colors['light'],
                                 fg=self.colors['muted'])
            info_label.pack(pady=5)
            
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
                    font=("Arial", 12, "bold"),
                    bg=self.colors['light'],
                    fg=self.colors['primary']).pack(pady=10)
            
            tk.Label(info_frame, text=f"📈 {rows:,} rows, {cols} columns",
                    font=("Arial", 10),
                    bg=self.colors['light'],
                    fg=self.colors['secondary']).pack()
            
            # Show first few columns
            columns_text = "Columns: " + ", ".join(self.csv_data.columns[:5].tolist())
            if len(self.csv_data.columns) > 5:
                columns_text += f"... (+{len(self.csv_data.columns)-5} more)"
            
            tk.Label(info_frame, text=columns_text,
                    font=("Arial", 9),
                    bg=self.colors['light'],
                    fg=self.colors['muted'],
                    wraplength=350).pack(pady=10)
            
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
        # Update labels
        self.disease_label.config(text=disease_name, font=("Arial", 12, "bold"))
        self.confidence_label.config(text=f"{confidence}%", font=("Arial", 12, "bold"))
        
        # Set colors based on disease
        if disease_name == "Healthy":
            color = self.colors['success']
        elif confidence >= 80:
            color = self.colors['danger']
        else:
            color = self.colors['warning']
        
        self.disease_label.config(fg=color)
        self.confidence_label.config(fg=color)
        
        # Update remedy text
        self.remedy_text.config(state="normal")
        self.remedy_text.delete(1.0, tk.END)
        
        formatted_remedy = f"🎯 ANALYSIS COMPLETE\n\n"
        formatted_remedy += f"Disease: {disease_name}\n"
        formatted_remedy += f"Confidence: {confidence}%\n\n"
        formatted_remedy += "TREATMENT:\n"
        formatted_remedy += "─" * 30 + "\n"
        formatted_remedy += remedy
        
        self.remedy_text.insert(1.0, formatted_remedy)
        self.remedy_text.config(state="disabled")
        
        messagebox.showinfo("Analysis Complete", 
                           f"Disease: {disease_name}\nConfidence: {confidence}%\n\nCheck results panel for treatment details.")
    
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