# main_gui.py

import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk
import pandas as pd
import joblib

from data_prep import clean_and_merge_data
from train_model import train_models
from analytics import generate_plots


class RealEstateMLGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Real Estate Market ML Dashboard")
        self.root.geometry("1400x900")
        self.root.configure(bg="#0f0f0f")

        self.results = None
        self.images = []

        self.configure_styles()
        self.create_widgets()

    def configure_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure(
            "Dark.TButton",
            font=("Segoe UI", 11, "bold"),
            padding=10,
            background="#1f1f1f",
            foreground="white",
            borderwidth=0
        )

        style.map(
            "Dark.TButton",
            background=[("active", "#2d2d2d")]
        )

        style.configure(
            "Dark.Horizontal.TProgressbar",
            troughcolor="#1a1a1a",
            background="#00c896",
            bordercolor="#1a1a1a",
            lightcolor="#00c896",
            darkcolor="#00c896"
        )

        style.configure(
            "Treeview",
            background="#181818",
            foreground="white",
            fieldbackground="#181818",
            rowheight=30,
            font=("Segoe UI", 10)
        )

        style.configure(
            "Treeview.Heading",
            background="#222222",
            foreground="white",
            font=("Segoe UI", 10, "bold")
        )

    def create_widgets(self):
        header_frame = tk.Frame(self.root, bg="#0f0f0f")
        header_frame.pack(fill="x", pady=10)

        title = tk.Label(
            header_frame,
            text="Real Estate Market ML Dashboard",
            font=("Segoe UI", 26, "bold"),
            bg="#0f0f0f",
            fg="#00c896"
        )
        title.pack()

        subtitle = tk.Label(
            header_frame,
            text="Machine Learning • Market Analytics • Investment Insights",
            font=("Segoe UI", 11),
            bg="#0f0f0f",
            fg="#bbbbbb"
        )
        subtitle.pack(pady=5)

        control_frame = tk.Frame(self.root, bg="#0f0f0f")
        control_frame.pack(fill="x", padx=15, pady=10)

        buttons = [
            ("Run Full Pipeline", self.run_pipeline),
            ("Show Top Picks", self.show_top_picks),
            ("Load Visualizations", self.load_visualizations),
            ("Export Logs", self.export_logs),
            ("System Status", self.show_status),
            ("Clear Log", self.clear_log)
        ]

        for i, (text, cmd) in enumerate(buttons):
            ttk.Button(
                control_frame,
                text=text,
                command=cmd,
                style="Dark.TButton"
            ).grid(row=0, column=i, padx=6, pady=5)

        self.progress = ttk.Progressbar(
            self.root,
            orient="horizontal",
            length=700,
            mode="indeterminate",
            style="Dark.Horizontal.TProgressbar"
        )
        self.progress.pack(pady=10)

        main_frame = tk.Frame(self.root, bg="#0f0f0f")
        main_frame.pack(fill="both", expand=True, padx=15, pady=10)

        left_panel = tk.Frame(main_frame, bg="#121212")
        left_panel.pack(side="left", fill="both", expand=False, padx=(0, 10))

        right_panel = tk.Frame(main_frame, bg="#121212")
        right_panel.pack(side="right", fill="both", expand=True)

        log_label = tk.Label(
            left_panel,
            text="Pipeline Logs",
            bg="#121212",
            fg="#00c896",
            font=("Segoe UI", 14, "bold")
        )
        log_label.pack(anchor="w", padx=10, pady=(10, 5))

        self.log_area = ScrolledText(
            left_panel,
            width=45,
            height=35,
            bg="#1a1a1a",
            fg="#e6e6e6",
            insertbackground="white",
            font=("Consolas", 10),
            relief="flat",
            borderwidth=10
        )
        self.log_area.pack(fill="both", expand=True, padx=10, pady=10)

        viz_label = tk.Label(
            right_panel,
            text="Analytics Visualizations",
            bg="#121212",
            fg="#00c896",
            font=("Segoe UI", 14, "bold")
        )
        viz_label.pack(anchor="w", padx=10, pady=(10, 5))

        self.canvas = tk.Canvas(
            right_panel,
            bg="#121212",
            highlightthickness=0,
            xscrollincrement=20
        )

        scrollbar = ttk.Scrollbar(
            right_panel,
            orient="vertical",
            command=self.canvas.yview
        )

        horizontal_scrollbar = ttk.Scrollbar(
            right_panel,
            orient="horizontal",
            command=self.canvas.xview
        )

        horizontal_scrollbar.configure(takefocus=0)

        self.scrollable_frame = tk.Frame(self.canvas, bg="#121212")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(
            yscrollcommand=scrollbar.set,
            xscrollcommand=horizontal_scrollbar.set
        )

        self.canvas.pack(side="left", fill="both", expand=True, padx=(0, 5))
        scrollbar.pack(side="right", fill="y")
        horizontal_scrollbar.pack(side="bottom", fill="x", padx=10, pady=(0, 8))

        self.canvas.bind_all("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind_all("<Shift-MouseWheel>", self.on_shift_mousewheel)

        # Linux trackpad horizontal scrolling
        # Some Tk versions do not support Button-6/Button-7.
        # Shift + Button-4/Button-5 provides broader compatibility.
        self.canvas.bind_all("<Shift-Button-4>", self.on_linux_horizontal_scroll)
        self.canvas.bind_all("<Shift-Button-5>", self.on_linux_horizontal_scroll)

        status_frame = tk.Frame(self.root, bg="#111111", height=25)
        status_frame.pack(fill="x", side="bottom")

        self.status_label = tk.Label(
            status_frame,
            text="Ready",
            bg="#111111",
            fg="#bbbbbb",
            anchor="w",
            font=("Segoe UI", 9)
        )
        self.status_label.pack(fill="x", padx=10)

    def on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def on_shift_mousewheel(self, event):
        delta = int(-1 * (event.delta / 120))
        self.canvas.xview_scroll(delta * 3, "units")

    def on_linux_horizontal_scroll(self, event):
        if event.num == 4:
            self.canvas.xview_scroll(-3, "units")
        elif event.num == 5:
            self.canvas.xview_scroll(3, "units")

    def set_status(self, text):
        self.status_label.config(text=text)
        self.root.update_idletasks()

    def log(self, message):
        self.log_area.insert(tk.END, message + "")
        self.log_area.see(tk.END)
        self.set_status(message)

    def clear_log(self):
        self.log_area.delete(1.0, tk.END)
        self.set_status("Logs cleared")

    def export_logs(self):
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text Files", "*.txt")]
            )

            if filename:
                with open(filename, "w") as f:
                    f.write(self.log_area.get(1.0, tk.END))

                messagebox.showinfo("Export Successful", "Logs exported successfully.")
                self.log("Logs exported successfully.")

        except Exception as e:
            self.log(f"ERROR: {str(e)}")

    def show_status(self):
        status_info = []

        files_to_check = [
            'data/final_cleaned_market_data.csv',
            'lean_stack_model.pkl',
            'scaler.pkl',
            'training_results.pkl'
        ]

        for file in files_to_check:
            exists = "Available" if os.path.exists(file) else "Missing"
            status_info.append(f"{file}: {exists}")

        messagebox.showinfo("System Status", "".join(status_info))
        self.log("Checked system status.")

    def run_pipeline(self):
        thread = threading.Thread(target=self.pipeline_task)
        thread.daemon = True
        thread.start()

    def pipeline_task(self):
        try:
            self.progress.start()
            self.log("=== Starting Real Estate ML Pipeline ===")

            data_path = 'data/final_cleaned_market_data.csv'
            results_path = 'training_results.pkl'

            if not os.path.exists(data_path):
                self.log("Cleaning and merging data...")
                clean_and_merge_data()
                self.log("Data preparation complete.")
            else:
                self.log(f"Using existing dataset: {data_path}")

            if os.path.exists(results_path):
                self.log("Loading cached training results...")
                self.results = joblib.load(results_path)
            else:
                self.log("Training machine learning models...")
                self.results = train_models(data_path=data_path)
                joblib.dump(self.results, results_path)
                self.log("Training complete and cached.")

            self.log("Generating visualizations...")
            generate_plots(self.results, data_path=data_path)
            self.log("Visualizations generated successfully.")

            self.load_visualizations()

            self.log("Pipeline completed successfully.")
            messagebox.showinfo("Success", "Pipeline completed successfully!")

        except Exception as e:
            self.log(f"ERROR: {str(e)}")
            messagebox.showerror("Error", str(e))

        finally:
            self.progress.stop()
            self.set_status("Ready")

    def show_top_picks(self):
        try:
            data_path = 'data/final_cleaned_market_data.csv'

            if not os.path.exists('lean_stack_model.pkl'):
                messagebox.showwarning("Missing Model", "Run the pipeline first.")
                return

            model = joblib.load('lean_stack_model.pkl')
            scaler = joblib.load('scaler.pkl')
            df = pd.read_csv(data_path)

            features = [
                'Typical_Monthly_Rent', 'CPI', 'Year', 'Month',
                'Days_To_Pending', 'ZHVF_Forecast', 'Mortgage_Payment',
                'Rent_Affordability_Ratio', 'Market_Heat_Index',
                'Rent_3mo_Avg', 'Pending_3mo_Avg', 'Rent_6mo_Growth',
                'Pending_Velocity_3mo', 'Is_Peak_Season'
            ]

            latest_date = df['Date'].max()
            current_market = df[df['Date'] == latest_date].copy()

            X_scaled = scaler.transform(current_market[features])
            current_market['Buy_Confidence'] = model.predict_proba(X_scaled)[:, 1]

            top_10 = current_market.sort_values(
                by='Buy_Confidence',
                ascending=False
            ).head(10)

            top_window = tk.Toplevel(self.root)
            top_window.title("Top Investment Picks")
            top_window.geometry("1000x500")
            top_window.configure(bg="#101010")

            tree = ttk.Treeview(
                top_window,
                columns=("Zip", "City", "State", "Heat", "Confidence"),
                show="headings"
            )

            headings = [
                ("Zip", "Zip Code"),
                ("City", "City"),
                ("State", "State"),
                ("Heat", "Market Heat"),
                ("Confidence", "Buy Confidence")
            ]

            for col, text in headings:
                tree.heading(col, text=text)
                tree.column(col, anchor="center", width=180)

            scrollbar = ttk.Scrollbar(top_window, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=scrollbar.set)

            for _, row in top_10.iterrows():
                tree.insert(
                    "",
                    tk.END,
                    values=(
                        row['RegionName'],
                        row['City'],
                        row['State'],
                        f"{row['Market_Heat_Index']:.2f}",
                        f"{row['Buy_Confidence'] * 100:.2f}%"
                    )
                )

            tree.pack(side="left", fill="both", expand=True, padx=10, pady=10)
            scrollbar.pack(side="right", fill="y")

            self.log("Displayed top 10 investment picks.")

        except Exception as e:
            self.log(f"ERROR: {str(e)}")
            messagebox.showerror("Error", str(e))

    def load_visualizations(self):
        try:
            for widget in self.scrollable_frame.winfo_children():
                widget.destroy()

            image_paths = [
                'visualizations/1_ROC_Curve_Comparison.png',
                'visualizations/2_Model_Comparison_CM.png',
                'visualizations/3_Feature_Importance_Comparison.png',
                'visualizations/4_Feature_Correlation.png'
            ]

            self.images.clear()

            for i, path in enumerate(image_paths):
                if os.path.exists(path):
                    frame = tk.Frame(self.scrollable_frame, bg="#1a1a1a")
                    frame.grid(row=i // 2, column=i % 2, padx=15, pady=15)

                    title = tk.Label(
                        frame,
                        text=os.path.basename(path).replace("_", " "),
                        bg="#1a1a1a",
                        fg="#00c896",
                        font=("Segoe UI", 11, "bold")
                    )
                    title.pack(pady=5)

                    img = Image.open(path)
                    img = img.resize((550, 320))
                    photo = ImageTk.PhotoImage(img)
                    self.images.append(photo)

                    label = tk.Label(frame, image=photo, bg="#1a1a1a")
                    label.pack()

            self.log("Visualizations loaded successfully.")

        except Exception as e:
            self.log(f"ERROR: {str(e)}")
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = RealEstateMLGUI(root)
    root.mainloop()