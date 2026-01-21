import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import pandas as pd
import os
from disease_detector import DiseaseDetector

class CropDiseaseDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("🌱 Crop Disease Detection System - Professional Edition")
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
            'primary': '#1e3a8a',      # Deep blue
            'secondary': '#64748b',     # Slate gray
            'accent': '#0ea5e9',       # Sky blue
            'success': '#059669',      # Emerald green
            'danger': '#dc2626',       # Red
            'warning': '#d97706',      # Amber
            'light': '#f1f5f9',       # Light slate
            'white': '#ffffff',       # Pure white
            'dark': '#0f172a',        # Dark slate
            'muted': '#94a3b8',       # Muted slate
            'card': '#ffffff',        # Card background
            'border': '#e2e8f0'       # Border color
        }
        
        # Configure ttk styles
        self.setup_styles()
        self.setup_ui()
    
    def setup_styles(self):
        """Configure modern ttk styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure button styles
        style.configure('Primary.TButton', 
                       background=self.colors['primary'],
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none',
                       padding=(20, 10))
        
        style.configure('Success.TButton',
                       background=self.colors['success'],
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none',
                       padding=(20, 12))
        
        style.configure('Warning.TButton',
                       background=self.colors['warning'],
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none',
                       padding=(20, 10))
    
    def setup_ui(self):
        # Modern header with gradient effect
        header_frame = tk.Frame(self.root, bg=self.colors['primary'], height=100)
        header_frame.pack(fill="x")
        header_frame.pack_propagate(False)
        
        # Header content
        header_content = tk.Frame(header_frame, bg=self.colors['primary'])
        header_content.pack(expand=True, fill='both')
        
        # Title section
        title_section = tk.Frame(header_content, bg=self.colors['primary'])
        title_section.pack(expand=True)
        
        # Main title
        main_title = tk.Label(title_section, 
                             text="🌱 Crop Disease Detection System",
                             font=("Segoe UI", 28, "bold"),
                             bg=self.colors['primary'],
                             fg=self.colors['white'])
        main_title.pack(pady=(15, 5))
        
        # Subtitle
        subtitle = tk.Label(title_section,
                           text="AI-Powered Agricultural Health Analysis",
                           font=("Segoe UI", 12),
                           bg=self.colors['primary'],
                           fg=self.colors['light'])
        subtitle.pack()
        
        # Main container with modern layout
        main_container = tk.Frame(self.root, bg=self.colors['light'])
        main_container.pack(expand=True, fill="both", padx=25, pady=25)
        
        # Left panel - Upload and preview section
        left_panel = tk.Frame(main_container, bg=self.colors['card'], 
                             relief="flat", bd=0)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 15))
        
        # Add subtle shadow effect
        shadow_frame = tk.Frame(main_container, bg=self.colors['border'], height=2)
        
        # Modern card styling
        left_panel.configure(highlightbackground=self.colors['border'], 
                           highlightthickness=1)
        
        # Upload section with modern design
        upload_section = tk.Frame(left_panel, bg=self.colors['card'])
        upload_section.pack(fill="x", padx=25, pady=25)
        
        # Section header
        upload_header = tk.Frame(upload_section, bg=self.colors['card'])
        upload_header.pack(fill="x", pady=(0, 20))
        
        upload_icon = tk.Label(upload_header, text="📁", 
                              font=("Segoe UI", 20), bg=self.colors['card'])
        upload_icon.pack(side="left")
        
        upload_title = tk.Label(upload_header, text="File Upload",
                               font=("Segoe UI", 16, "bold"),
                               bg=self.colors['card'],
                               fg=self.colors['primary'])
        upload_title.pack(side="left", padx=(10, 0))
        
        upload_desc = tk.Label(upload_section, 
                              text="Select an image file or CSV data for analysis",
                              font=("Segoe UI", 10),
                              bg=self.colors['card'],
                              fg=self.colors['muted'])
        upload_desc.pack(anchor="w", pady=(0, 15))
        
        # Modern button container
        btn_container = tk.Frame(upload_section, bg=self.colors['card'])
        btn_container.pack(fill="x")
        
        # Image upload button with modern styling
        self.upload_img_btn = tk.Button(btn_container, 
                                       text="📷  Upload Image",
                                       command=self.upload_image,
                                       font=("Segoe UI", 11, "bold"),
                                       bg=self.colors['accent'],
                                       fg=self.colors['white'],
                                       relief="flat",
                                       padx=25, pady=12,
                                       cursor="hand2",
                                       activebackground=self.colors['primary'])
        self.upload_img_btn.pack(side="left", padx=(0, 15))
        
        # CSV upload button
        self.upload_csv_btn = tk.Button(btn_container,
                                       text="📊  Upload CSV",
                                       command=self.upload_csv,
                                       font=("Segoe UI", 11, "bold"),
                                       bg=self.colors['warning'],
                                       fg=self.colors['white'],
                                       relief="flat",
                                       padx=25, pady=12,
                                       cursor="hand2",
                                       activebackground=self.colors['danger'])
        self.upload_csv_btn.pack(side="left")
        
        # Preview section with modern card design
        preview_section = tk.Frame(left_panel, bg=self.colors['card'])
        preview_section.pack(fill="both", expand=True, padx=25, pady=(15, 25))
        
        # Preview header
        preview_header = tk.Frame(preview_section, bg=self.colors['card'])
        preview_header.pack(fill="x", pady=(0, 15))
        
        preview_icon = tk.Label(preview_header, text="🖼️",
                               font=("Segoe UI", 20), bg=self.colors['card'])
        preview_icon.pack(side="left")
        
        preview_title = tk.Label(preview_header, text="File Preview",
                                font=("Segoe UI", 16, "bold"),
                                bg=self.colors['card'],
                                fg=self.colors['primary'])
        preview_title.pack(side="left", padx=(10, 0))
        
        # Modern display area with rounded corners effect
        display_container = tk.Frame(preview_section, bg=self.colors['light'],
                                   relief="flat", bd=0)
        display_container.pack(fill="both", expand=True)
        
        # Inner display frame
        self.display_frame = tk.Frame(display_container, bg=self.colors['light'],
                                     highlightbackground=self.colors['border'],
                                     highlightthickness=2)
        self.display_frame.pack(fill="both", expand=True, padx=2, pady=2)
        
        # Placeholder content
        placeholder_frame = tk.Frame(self.display_frame, bg=self.colors['light'])
        placeholder_frame.pack(expand=True)
        
        self.display_label = tk.Label(placeholder_frame,
                                     text="📷 📊\n\nNo file selected\n\nUpload an image or CSV file to begin analysis",
                                     bg=self.colors['light'],
                                     font=("Segoe UI", 13),
                                     fg=self.colors['muted'],
                                     justify="center")
        self.display_label.pack(expand=True)
        
        # Right panel - Analysis and results section
        right_panel = tk.Frame(main_container, bg=self.colors['card'],
                              relief="flat", bd=0, width=420)
        right_panel.pack(side="right", fill="y")
        right_panel.pack_propagate(False)
        
        # Modern card styling for right panel
        right_panel.configure(highlightbackground=self.colors['border'],
                             highlightthickness=1)
        
        # Analysis control section
        analysis_section = tk.Frame(right_panel, bg=self.colors['card'])
        analysis_section.pack(fill="x", padx=25, pady=25)
        
        # Analysis header
        analysis_header = tk.Frame(analysis_section, bg=self.colors['card'])
        analysis_header.pack(fill="x", pady=(0, 20))
        
        analysis_icon = tk.Label(analysis_header, text="🔬",
                                font=("Segoe UI", 20), bg=self.colors['card'])
        analysis_icon.pack(side="left")
        
        analysis_title = tk.Label(analysis_header, text="Disease Analysis",
                                 font=("Segoe UI", 16, "bold"),
                                 bg=self.colors['card'],
                                 fg=self.colors['primary'])
        analysis_title.pack(side="left", padx=(10, 0))
        
        analysis_desc = tk.Label(analysis_section,
                                text="Click below to start AI-powered disease detection",
                                font=("Segoe UI", 10),
                                bg=self.colors['card'],
                                fg=self.colors['muted'])
        analysis_desc.pack(anchor="w", pady=(0, 20))
        
        # Modern analyze button
        self.detect_btn = tk.Button(analysis_section,
                                   text="🔍  Start Analysis",
                                   command=self.detect_disease,
                                   font=("Segoe UI", 12, "bold"),
                                   bg=self.colors['success'],
                                   fg=self.colors['white'],
                                   relief="flat",
                                   padx=40, pady=15,
                                   cursor="hand2",
                                   state="disabled",
                                   activebackground=self.colors['primary'])
        self.detect_btn.pack(fill="x")
        
        # Results section with modern design
        results_section = tk.Frame(right_panel, bg=self.colors['card'])
        results_section.pack(fill="both", expand=True, padx=25, pady=(15, 25))
        
        # Results header
        results_header = tk.Frame(results_section, bg=self.colors['card'])
        results_header.pack(fill="x", pady=(0, 20))
        
        results_icon = tk.Label(results_header, text="📊",
                               font=("Segoe UI", 20), bg=self.colors['card'])
        results_icon.pack(side="left")
        
        results_title = tk.Label(results_header, text="Detection Results",
                                font=("Segoe UI", 16, "bold"),
                                bg=self.colors['card'],
                                fg=self.colors['primary'])
        results_title.pack(side="left", padx=(10, 0))
        
        # Results container with subtle border
        results_container = tk.Frame(results_section, bg=self.colors['light'],
                                   highlightbackground=self.colors['border'],
                                   highlightthickness=1)
        results_container.pack(fill="both", expand=True, pady=(0, 0))
        
        # Disease result card
        disease_card = tk.Frame(results_container, bg=self.colors['white'])
        disease_card.pack(fill="x", padx=20, pady=(20, 15))
        
        disease_header = tk.Frame(disease_card, bg=self.colors['white'])
        disease_header.pack(fill="x", pady=(0, 8))
        
        tk.Label(disease_header, text="🦠", font=("Segoe UI", 16),
                bg=self.colors['white']).pack(side="left")
        tk.Label(disease_header, text="Disease Type",
                font=("Segoe UI", 11, "bold"),
                bg=self.colors['white'],
                fg=self.colors['secondary']).pack(side="left", padx=(8, 0))
        
        self.disease_label = tk.Label(disease_card, text="Awaiting analysis...",
                                     font=("Segoe UI", 14, "bold"),
                                     bg=self.colors['white'],
                                     fg=self.colors['muted'])
        self.disease_label.pack(anchor="w", pady=(0, 5))
        
        # Confidence result card
        confidence_card = tk.Frame(results_container, bg=self.colors['white'])
        confidence_card.pack(fill="x", padx=20, pady=15)
        
        confidence_header = tk.Frame(confidence_card, bg=self.colors['white'])
        confidence_header.pack(fill="x", pady=(0, 8))
        
        tk.Label(confidence_header, text="📈", font=("Segoe UI", 16),
                bg=self.colors['white']).pack(side="left")
        tk.Label(confidence_header, text="Confidence Level",
                font=("Segoe UI", 11, "bold"),
                bg=self.colors['white'],
                fg=self.colors['secondary']).pack(side="left", padx=(8, 0))
        
        confidence_display = tk.Frame(confidence_card, bg=self.colors['white'])
        confidence_display.pack(fill="x", pady=(0, 10))
        
        self.confidence_label = tk.Label(confidence_display, text="0%",
                                        font=("Segoe UI", 14, "bold"),
                                        bg=self.colors['white'],
                                        fg=self.colors['muted'])
        self.confidence_label.pack(side="left")
        
        # Modern progress bar
        self.progress_var = tk.DoubleVar()
        progress_frame = tk.Frame(confidence_card, bg=self.colors['white'])
        progress_frame.pack(fill="x")
        
        self.progress_bar = ttk.Progressbar(progress_frame,
                                           variable=self.progress_var,
                                           maximum=100,
                                           length=300,
                                           mode='determinate',
                                           style='TProgressbar')
        self.progress_bar.pack(fill="x")
        
        # Treatment recommendation card
        remedy_card = tk.Frame(results_container, bg=self.colors['white'])
        remedy_card.pack(fill="both", expand=True, padx=20, pady=(15, 20))
        
        remedy_header = tk.Frame(remedy_card, bg=self.colors['white'])
        remedy_header.pack(fill="x", pady=(0, 10))
        
        tk.Label(remedy_header, text="💊", font=("Segoe UI", 16),
                bg=self.colors['white']).pack(side="left")
        tk.Label(remedy_header, text="Treatment Recommendations",
                font=("Segoe UI", 11, "bold"),
                bg=self.colors['white'],
                fg=self.colors['secondary']).pack(side="left", padx=(8, 0))
        
        # Modern text area with better styling
        text_container = tk.Frame(remedy_card, bg=self.colors['light'],
                                 highlightbackground=self.colors['border'],
                                 highlightthickness=1)
        text_container.pack(fill="both", expand=True)
        
        self.remedy_text = tk.Text(text_container,
                                  height=7,
                                  wrap="word",
                                  font=("Segoe UI", 10),
                                  bg=self.colors['light'],
                                  fg=self.colors['dark'],
                                  relief="flat",
                                  bd=0,
                                  padx=15,
                                  pady=12,
                                  selectbackground=self.colors['accent'],
                                  selectforeground=self.colors['white'])
        
        scrollbar = ttk.Scrollbar(text_container,
                                 orient="vertical",
                                 command=self.remedy_text.yview)
        self.remedy_text.configure(yscrollcommand=scrollbar.set)
        
        self.remedy_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Initialize with placeholder text
        self.remedy_text.insert(1.0, "Upload a file and run analysis to receive personalized treatment recommendations based on detected conditions.")
        self.remedy_text.config(state="disabled")
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Crop Image for Analysis",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.image_path = file_path
            self.csv_path = None
            self.file_type = 'image'
            self.display_image()
            
            # Update UI state
            self.detect_btn.config(state="normal", bg=self.colors['success'])
            self.upload_img_btn.config(text="✅  Image Loaded", bg=self.colors['success'])
            self.upload_csv_btn.config(text="📊  Upload CSV", bg=self.colors['warning'])
            self.reset_results()
            
            # Show success message
            filename = os.path.basename(file_path)
            messagebox.showinfo("✅ Success", f"Image '{filename}' loaded successfully!\nReady for analysis.")
    
    def upload_csv(self):
        file_path = filedialog.askopenfilename(
            title="Select CSV Data File for Analysis",
            filetypes=[
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx *.xls"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Load CSV with better error handling
                if file_path.endswith(('.xlsx', '.xls')):
                    self.csv_data = pd.read_excel(file_path)
                else:
                    self.csv_data = pd.read_csv(file_path)
                
                self.csv_path = file_path
                self.image_path = None
                self.file_type = 'csv'
                self.display_csv()
                
                # Update UI state
                self.detect_btn.config(state="normal", bg=self.colors['success'])
                self.upload_csv_btn.config(text="✅  CSV Loaded", bg=self.colors['success'])
                self.upload_img_btn.config(text="📷  Upload Image", bg=self.colors['accent'])
                self.reset_results()
                
                # Show success message
                filename = os.path.basename(file_path)
                rows, cols = self.csv_data.shape
                messagebox.showinfo("✅ Success", 
                                   f"CSV '{filename}' loaded successfully!\n\nData: {rows} rows, {cols} columns\nReady for analysis.")
                
            except Exception as e:
                messagebox.showerror("❌ Error", f"Failed to load file: {str(e)}\n\nPlease ensure the file is a valid CSV or Excel format.")
    
    def display_image(self):
        try:
            # Load and process image for display
            image = Image.open(self.image_path)
            original_size = image.size
            
            # Calculate optimal display size maintaining aspect ratio
            max_width, max_height = 500, 400
            image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Clear the display frame and add image
            for widget in self.display_frame.winfo_children():
                widget.destroy()
            
            # Create image display with info
            img_container = tk.Frame(self.display_frame, bg=self.colors['light'])
            img_container.pack(expand=True)
            
            # Image label
            img_label = tk.Label(img_container, image=photo, bg=self.colors['light'])
            img_label.pack(pady=10)
            img_label.image = photo  # Keep reference
            
            # Image info
            filename = os.path.basename(self.image_path)
            info_text = f"📷 {filename}\nOriginal: {original_size[0]}×{original_size[1]} pixels"
            info_label = tk.Label(img_container, text=info_text,
                                 font=("Segoe UI", 9),
                                 bg=self.colors['light'],
                                 fg=self.colors['muted'])
            info_label.pack(pady=(5, 10))
            
        except Exception as e:
            messagebox.showerror("❌ Error", f"Failed to display image: {str(e)}")
            self.upload_img_btn.config(text="📷  Upload Image", bg=self.colors['accent'])
    
    def display_csv(self):
        try:
            # Clear the display frame
            for widget in self.display_frame.winfo_children():
                widget.destroy()
            
            # Create CSV preview container
            csv_container = tk.Frame(self.display_frame, bg=self.colors['light'])
            csv_container.pack(expand=True, fill="both", padx=20, pady=20)
            
            # CSV header info
            filename = os.path.basename(self.csv_path)
            rows, cols = self.csv_data.shape
            
            header_frame = tk.Frame(csv_container, bg=self.colors['white'],
                                   highlightbackground=self.colors['border'],
                                   highlightthickness=1)
            header_frame.pack(fill="x", pady=(0, 15))
            
            # File info
            tk.Label(header_frame, text=f"📊 {filename}",
                    font=("Segoe UI", 12, "bold"),
                    bg=self.colors['white'],
                    fg=self.colors['primary']).pack(pady=10)
            
            stats_frame = tk.Frame(header_frame, bg=self.colors['white'])
            stats_frame.pack(pady=(0, 10))
            
            tk.Label(stats_frame, text=f"📈 {rows:,} rows",
                    font=("Segoe UI", 10),
                    bg=self.colors['white'],
                    fg=self.colors['secondary']).pack(side="left", padx=10)
            
            tk.Label(stats_frame, text=f"📋 {cols} columns",
                    font=("Segoe UI", 10),
                    bg=self.colors['white'],
                    fg=self.colors['secondary']).pack(side="left", padx=10)
            
            # Column preview
            columns_frame = tk.Frame(csv_container, bg=self.colors['white'],
                                   highlightbackground=self.colors['border'],
                                   highlightthickness=1)
            columns_frame.pack(fill="both", expand=True)
            
            tk.Label(columns_frame, text="📋 Column Preview",
                    font=("Segoe UI", 11, "bold"),
                    bg=self.colors['white'],
                    fg=self.colors['primary']).pack(pady=(10, 5))
            
            # Scrollable column list
            columns_text = tk.Text(columns_frame, height=8, wrap="word",
                                  font=("Segoe UI", 9),
                                  bg=self.colors['light'],
                                  fg=self.colors['dark'],
                                  relief="flat", bd=0,
                                  padx=10, pady=5)
            
            # Add columns with sample data
            column_info = ""
            for i, col in enumerate(self.csv_data.columns[:15], 1):
                sample_val = str(self.csv_data[col].iloc[0]) if len(self.csv_data) > 0 else "N/A"
                if len(sample_val) > 30:
                    sample_val = sample_val[:30] + "..."
                column_info += f"{i:2d}. {col}\n    Sample: {sample_val}\n\n"
            
            if len(self.csv_data.columns) > 15:
                column_info += f"... and {len(self.csv_data.columns) - 15} more columns"
            
            columns_text.insert(1.0, column_info)
            columns_text.config(state="disabled")
            columns_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))
            
        except Exception as e:
            messagebox.showerror("❌ Error", f"Failed to display CSV preview: {str(e)}")
            self.upload_csv_btn.config(text="📊  Upload CSV", bg=self.colors['warning'])
    
    def detect_disease(self):
        if not self.image_path and not self.csv_path:
            messagebox.showwarning("⚠️ Warning", "Please upload a file first!")
            return
        
        if self.file_type == 'image':
            self.analyze_image()
        elif self.file_type == 'csv':
            self.analyze_csv()
    
    def analyze_image(self):
        try:
            # Use disease detector for analysis
            disease_name, confidence, remedy, marked_image = self.detector.analyze_image(self.image_path)
            
            # Display marked image
            marked_image.thumbnail((400, 300), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(marked_image)
            self.display_label.config(image=photo, text="")
            self.display_label.image = photo
            
            self.display_results(disease_name, confidence, remedy, "Image Analysis")
            
        except Exception as e:
            messagebox.showerror("❌ Error", f"Image analysis failed: {str(e)}")
    
    def analyze_csv(self):
        try:
            # Use disease detector for CSV analysis
            disease_name, confidence, remedy = self.detector.analyze_csv(self.csv_data)
            
            self.display_results(disease_name, confidence, remedy, "CSV Data Analysis")
            
        except Exception as e:
            messagebox.showerror("❌ Error", str(e))
    
    def display_results(self, disease_name, confidence, remedy, analysis_type):
        # Update disease label with modern styling
        self.disease_label.config(text=disease_name, font=("Segoe UI", 14, "bold"))
        self.confidence_label.config(text=f"{confidence}%", font=("Segoe UI", 14, "bold"))
        
        # Animate progress bar
        self.animate_progress(confidence)
        
        # Set colors based on disease type and confidence
        if disease_name == "Healthy":
            color = self.colors['success']
            status_emoji = "✅"
        elif confidence >= 80:
            color = self.colors['danger']
            status_emoji = "⚠️"
        else:
            color = self.colors['warning']
            status_emoji = "❓"
        
        # Update colors
        self.disease_label.config(fg=color)
        self.confidence_label.config(fg=color)
        
        # Update remedy text with better formatting
        self.remedy_text.config(state="normal")
        self.remedy_text.delete(1.0, tk.END)
        
        # Format remedy text nicely
        formatted_remedy = f"{status_emoji} ANALYSIS RESULTS\n\n"
        formatted_remedy += f"Disease Detected: {disease_name}\n"
        formatted_remedy += f"Confidence Level: {confidence}%\n\n"
        formatted_remedy += "RECOMMENDED TREATMENT:\n"
        formatted_remedy += "─" * 40 + "\n"
        formatted_remedy += remedy
        
        if analysis_type == "Image Analysis":
            formatted_remedy += "\n\n📝 NOTE: Analysis based on visual inspection of uploaded image."
        else:
            formatted_remedy += "\n\n📝 NOTE: Analysis based on provided CSV data."
        
        self.remedy_text.insert(1.0, formatted_remedy)
        self.remedy_text.config(state="disabled")
        
        # Show comprehensive success message
        result_msg = f"🎯 Analysis Complete!\n\n"
        result_msg += f"Type: {analysis_type}\n"
        result_msg += f"Disease: {disease_name}\n"
        result_msg += f"Confidence: {confidence}%\n\n"
        result_msg += "✅ Detailed treatment recommendations are now available in the results panel."
        
        messagebox.showinfo("Analysis Complete", result_msg)
    
    def animate_progress(self, target_value):
        """Animate progress bar to target value"""
        current = self.progress_var.get()
        if current < target_value:
            self.progress_var.set(current + 2)
            self.root.after(50, lambda: self.animate_progress(target_value))
    
    def reset_results(self):
        """Reset detection results with modern styling"""
        self.disease_label.config(text="Awaiting analysis...", 
                                 fg=self.colors['muted'],
                                 font=("Segoe UI", 14, "bold"))
        self.confidence_label.config(text="0%", 
                                    fg=self.colors['muted'],
                                    font=("Segoe UI", 14, "bold"))
        self.progress_var.set(0)
        
        # Reset remedy text
        self.remedy_text.config(state="normal")
        self.remedy_text.delete(1.0, tk.END)
        placeholder_text = "🔬 READY FOR ANALYSIS\n\n"
        placeholder_text += "Upload a file and click 'Start Analysis' to begin AI-powered disease detection.\n\n"
        placeholder_text += "📋 SUPPORTED FILES:\n"
        placeholder_text += "• Images: JPG, PNG, BMP, TIFF\n"
        placeholder_text += "• Data: CSV, Excel files\n\n"
        placeholder_text += "🎯 Get instant results with treatment recommendations!"
        
        self.remedy_text.insert(1.0, placeholder_text)
        self.remedy_text.config(state="disabled")

def main():
    """Launch the Crop Disease Detection System"""
    root = tk.Tk()
    
    # Set window icon (if available)
    try:
        # You can add an icon file here
        # root.iconbitmap('icon.ico')
        pass
    except:
        pass
    
    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")
    
    # Initialize application
    app = CropDiseaseDetector(root)
    
    # Start the application
    root.mainloop()

if __name__ == "__main__":
    main()