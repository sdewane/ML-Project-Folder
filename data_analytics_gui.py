# ══════════════════════════════════════════════════════════════════════════════
#  data_analytics_gui.py  —  Tkinter GUI wrapper for data_analytics.py
#  Run:  python data_analytics_gui.py
#  Keeps every original function intact; only adds GUI layer below.
# ══════════════════════════════════════════════════════════════════════════════

# ── Original imports (unchanged) ──────────────────────────────────────────────
# Imprt necessary libraries for data manipulation, visualization, and machine learning.
# %pip install -q numpy pandas seaborn matplotlib scikit-learn

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")          # <-- added: must be set before pyplot import
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
)

# ── GUI-only imports (added) ──────────────────────────────────────────────────
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
import io

# ── Original config (unchanged) ───────────────────────────────────────────────
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: f"{x:.3f}")
sns.set_theme(style="darkgrid")

plt.rcParams.update({
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
})

RANDOM_STATE = 42
CSV_PATH = "data/housing.csv"       # default; overridden by GUI
TARGET_COL = "median_house_value"   # default; overridden by GUI

# ── Original functions (unchanged — df passed as parameter where needed) ──────

def data_info(df):
    print("Dataset shape:", df.shape)
    print(df.head())
    print("Column data: ")
    print(df.columns)
    print(df.info())

#data_info(df)
#^^^^^^^^^^^^^^^ uncomment to see basic data info

def missing_data(df):
    print("\nMissing values per column: ")
    print(df.isna().sum())

#missing_data(df)
#^^^^^^^^^^^^^^^^^^uncomment to print out how much data is missing per columns

def find_encoded_data(df):
    for col in df.columns:
        print(df[col].value_counts().head(20))

#find_encoded_data(df)
#^^^^^^^^^^^^^^^^^^^^^^^uncomment to see if there are any encoded missing value entries per column

def find_duplicates(df):
    duplicate_mask = df.duplicated()
    num_duplicates = duplicate_mask.sum()
    print("Num of duplicated rows: ", num_duplicates)
    #optional
    #df = df.drop_duplicates()
    #print("Shape after dropping duplicates: ", df.shape)

#find_duplicates(df)
#^^^^^^^^^^^^^^^^^^^^^uncomment to find ducplicates in data and optionaly clean up duplicates

#print(df[num_cols].describe().T)
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^uncomment to see data statistics transposed

def countplot_cattegorical_columns(categorical_cols, df):        # df added as param
    for col in categorical_cols:
        plt.figure(figsize=(10,3))
        sns.countplot(x=col, data=df)
        plt.title(f"Distribution of {col}")
        print(df[col].value_counts())
        plt.show()

#countplot_cattegorical_columns(categorical_cols)
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^uncomment to see plot of categorical data and exact numbers

def target_col_distribution(target, df):
        plt.figure(figsize=(6,4))
        sns.histplot(df[target], bins=40, kde=True)
        plt.title(f"Target distribution: "+ target)
        plt.xlabel(""+ target)
        print(df[target].value_counts())
        plt.show()

#target_col_distribution(TARGET_COL,df)
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^uncomment to see graph and exact numbers

def all_num_column_distribution(num_cols, df):
    fig, axes = plt.subplots(3, 3, figsize=(8, 6))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        sns.histplot(df[col], kde=True, ax=axes[i])
        axes[i].set_title(col, fontsize=8)
    plt.tight_layout()
    plt.show()

#all_num_column_distribution(num_cols,df)
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^uncomment to see graph

def boxplot_visial(num_cols, df):
    fig, axes = plt.subplots(3, 3, figsize=(8, 6))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        sns.boxplot(x=df[col], ax=axes[i])
        axes[i].set_title(col, fontsize=8)
        axes[i].set_xlabel("")
    plt.tight_layout()
    plt.show()

#boxplot_visial(num_cols,df)
#^^^^^^^^^^^^^^^^^^^^^^^^^^^uncomment to see box plot to detect outliers

def id_corrolated_col_features(num_cols, df):
    plt.figure(figsize=(10, 5))
    sns.heatmap(
        df[num_cols].corr(),
        annot=True,
        cmap="coolwarm",
        center=0
    )
    plt.title("Correlation Heatmap")
    plt.show()

#id_corrolated_col_features(num_cols,df)
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^Uncomment to see heat graph,high correlation means using a linear model, other models would require highly correlated columns to be dropped

def corr_wth_trg_col(target, num_cols, df):
    corr_wth_taget = df[num_cols].corr()[target].sort_values(ascending=False)
    print("\ncorrelation with target: ")
    print(corr_wth_taget)

#corr_wth_trg_col(TARGET_COL, num_cols,df)
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^uncomment to see correlation of numerical datapoints and target value


# ══════════════════════════════════════════════════════════════════════════════
#  GUI  (everything below is new)
# ══════════════════════════════════════════════════════════════════════════════

# ── Colour palette ────────────────────────────────────────────────────────────
BG        = "#0f0f17"
PANEL     = "#16161f"
CARD      = "#1e1e2e"
BORDER    = "#2a2a3d"
FG        = "#e2e2f0"
FG_DIM    = "#7f7f9f"
ACCENT    = "#7c6af7"   # violet
GREEN     = "#4ecca3"
RED       = "#f05d7a"
YELLOW    = "#f5c842"
MONO_FONT = ("Courier New", 10)
UI_FONT   = ("Helvetica", 10)
BOLD_FONT = ("Helvetica", 10, "bold")


class _ScrollableFrame(tk.Frame):
    """Vertically scrollable container for mixed widgets (text + plots)."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=BG, **kwargs)
        self._canvas = tk.Canvas(self, bg=BG, highlightthickness=0)
        sb = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self.inner = tk.Frame(self._canvas, bg=BG)

        self.inner.bind(
            "<Configure>",
            lambda _: self._canvas.configure(scrollregion=self._canvas.bbox("all"))
        )
        self._win = self._canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self._canvas.configure(yscrollcommand=sb.set)
        self._canvas.bind("<Configure>", self._on_canvas_resize)

        sb.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)

        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self._canvas.bind_all(seq, self._scroll)

    def _on_canvas_resize(self, event):
        self._canvas.itemconfig(self._win, width=event.width)

    def _scroll(self, event):
        if event.num == 4:
            self._canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self._canvas.yview_scroll(1, "units")
        else:
            self._canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def scroll_to_bottom(self):
        self._canvas.update_idletasks()
        self._canvas.yview_moveto(1.0)


class DataAnalyticsApp:
    """Main application window."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Data Analytics — Housing Explorer")
        self.root.geometry("1280x820")
        self.root.minsize(900, 600)
        self.root.configure(bg=BG)

        self.df          = None
        self.num_cols    = []
        self.cat_cols    = []
        self._csv_var    = tk.StringVar(value=CSV_PATH)
        self._target_var = tk.StringVar(value=TARGET_COL)

        self._patch_plt_show()
        self._apply_style()
        self._build_ui()

    # ── plt.show() override ───────────────────────────────────────────────────
    def _patch_plt_show(self):
        """Redirect plt.show() so every figure embeds in the Plots tab."""
        app = self

        def _embedded_show():
            fig = plt.gcf()
            app._embed_figure(fig)
            plt.close(fig)

        plt.show = _embedded_show

    # ── ttk styling ───────────────────────────────────────────────────────────
    def _apply_style(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook",          background=BG,    borderwidth=0)
        style.configure("TNotebook.Tab",      background=PANEL, foreground=FG_DIM,
                         padding=[14, 6],     font=BOLD_FONT)
        style.map("TNotebook.Tab",
                  background=[("selected", CARD)],
                  foreground=[("selected", FG)])
        style.configure("TScrollbar", background=BORDER, troughcolor=BG,
                         arrowcolor=FG_DIM, borderwidth=0)
        style.configure("TSeparator", background=BORDER)

    # ── UI construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        self._build_topbar()

        paned = tk.PanedWindow(self.root, orient="horizontal",
                                bg=BORDER, sashwidth=3, sashrelief="flat")
        paned.pack(fill="both", expand=True, padx=0, pady=0)

        paned.add(self._build_sidebar(paned), minsize=200)
        paned.add(self._build_output(paned),  minsize=500)

    def _build_topbar(self):
        bar = tk.Frame(self.root, bg=PANEL, pady=10, padx=14)
        bar.pack(fill="x", side="top")

        # App title
        tk.Label(bar, text="◈  DATA ANALYTICS", bg=PANEL, fg=ACCENT,
                 font=("Helvetica", 12, "bold")).pack(side="left", padx=(0, 24))

        # CSV path
        tk.Label(bar, text="CSV", bg=PANEL, fg=FG_DIM, font=UI_FONT).pack(side="left")
        self._path_entry = tk.Entry(
            bar, textvariable=self._csv_var, width=38,
            bg=CARD, fg=FG, insertbackground=FG, relief="flat",
            font=UI_FONT, bd=0
        )
        self._path_entry.pack(side="left", padx=(6, 4), ipady=4)
        self._btn(bar, "Browse", self._browse, BORDER, FG).pack(side="left", padx=(0, 18))

        # Target column
        tk.Label(bar, text="Target", bg=PANEL, fg=FG_DIM, font=UI_FONT).pack(side="left")
        tk.Entry(
            bar, textvariable=self._target_var, width=22,
            bg=CARD, fg=FG, insertbackground=FG, relief="flat",
            font=UI_FONT, bd=0
        ).pack(side="left", padx=(6, 18), ipady=4)

        self._btn(bar, "⬆  Load Data", self._load_data, GREEN, BG).pack(side="left")

        # Status badge
        self._status = tk.Label(bar, text="No data loaded", bg=PANEL,
                                 fg=FG_DIM, font=UI_FONT)
        self._status.pack(side="left", padx=18)

    def _build_sidebar(self, parent):
        frame = tk.Frame(parent, bg=PANEL, padx=10, pady=10)

        tk.Label(frame, text="TOOLS", bg=PANEL, fg=FG_DIM,
                 font=("Helvetica", 8, "bold")).pack(anchor="w", pady=(0, 8))

        tools = [
            ("📋  Dataset Info",           self._run_data_info),
            ("❓  Missing Values",          self._run_missing),
            ("🔍  Encoded Value Check",     self._run_encoded),
            ("🔁  Find Duplicates",         self._run_duplicates),
            ("📊  Categorical Plots",       self._run_cat_plots),
            ("🎯  Target Distribution",     self._run_target_dist),
            ("📈  Numeric Distributions",   self._run_num_dist),
            ("📦  Box Plots",               self._run_boxplots),
            ("🔥  Correlation Heatmap",     self._run_corr_heatmap),
            ("🎯  Corr. with Target",       self._run_corr_target),
        ]

        for label, cmd in tools:
            b = tk.Button(
                frame, text=label, command=cmd, anchor="w",
                bg=CARD, fg=FG, activebackground=BORDER, activeforeground=FG,
                relief="flat", font=UI_FONT, padx=10, pady=7,
                cursor="hand2", bd=0
            )
            b.pack(fill="x", pady=2)
            b.bind("<Enter>", lambda e, w=b: w.config(bg=BORDER))
            b.bind("<Leave>", lambda e, w=b: w.config(bg=CARD))

        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=12)

        clear_btn = tk.Button(
            frame, text="🧹  Clear Output", command=self._clear_output,
            bg=RED, fg="#fff", activebackground="#c0364e",
            relief="flat", font=BOLD_FONT, padx=10, pady=7, cursor="hand2"
        )
        clear_btn.pack(fill="x")

        return frame

    def _build_output(self, parent):
        frame = tk.Frame(parent, bg=BG)

        self._notebook = ttk.Notebook(frame)
        self._notebook.pack(fill="both", expand=True, padx=6, pady=6)

        # Console tab
        console_frame = tk.Frame(self._notebook, bg=BG)
        self._notebook.add(console_frame, text="  Console  ")
        self._console = scrolledtext.ScrolledText(
            console_frame, bg=CARD, fg=GREEN, insertbackground=FG,
            font=MONO_FONT, relief="flat", wrap="word", padx=12, pady=10
        )
        self._console.pack(fill="both", expand=True)
        self._console.tag_configure("err",  foreground=RED)
        self._console.tag_configure("warn", foreground=YELLOW)
        self._console.tag_configure("head", foreground=ACCENT, font=("Helvetica", 10, "bold"))

        # Plots tab
        plots_frame = tk.Frame(self._notebook, bg=BG)
        self._notebook.add(plots_frame, text="  Plots  ")
        self._plot_scroll = _ScrollableFrame(plots_frame)
        self._plot_scroll.pack(fill="both", expand=True)

        return frame

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _btn(parent, text, cmd, bg, fg):
        return tk.Button(
            parent, text=text, command=cmd, bg=bg, fg=fg,
            activebackground=bg, activeforeground=fg,
            relief="flat", font=BOLD_FONT, padx=12, pady=4, cursor="hand2"
        )

    def _log(self, text, tag=None):
        self._console.insert(tk.END, text + "\n", tag or "")
        self._console.see(tk.END)
        self._notebook.select(0)

    def _embed_figure(self, fig):
        """Attach a matplotlib figure into the scrollable Plots tab."""
        wrapper = tk.Frame(self._plot_scroll.inner, bg=CARD,
                            highlightbackground=BORDER, highlightthickness=1,
                            padx=4, pady=4)
        wrapper.pack(fill="x", padx=10, pady=6)

        canvas = FigureCanvasTkAgg(fig, master=wrapper)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both")

        self._plot_scroll.scroll_to_bottom()
        self._notebook.select(1)

    def _require_data(self) -> bool:
        if self.df is None:
            self._log("[!] Load a dataset first.", "warn")
            return False
        return True

    def _run(self, func, *args):
        """Execute a function, capturing stdout into the Console tab."""
        old_stdout = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            func(*args)
        except Exception as exc:
            sys.stdout = old_stdout
            self._log(f"[ERROR] {exc}", "err")
            return
        sys.stdout = old_stdout
        captured = buf.getvalue()
        if captured.strip():
            self._log(captured)

    # ── Data loading ──────────────────────────────────────────────────────────
    def _browse(self):
        path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if path:
            self._csv_var.set(path)

    def _load_data(self):
        path = self._csv_var.get()
        try:
            self.df       = pd.read_csv(path)
            self.num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            self.cat_cols = self.df.select_dtypes(include=["object"]).columns.tolist()
            msg = f"✓  {self.df.shape[0]:,} rows × {self.df.shape[1]} columns"
            self._status.config(text=msg, fg=GREEN)
            self._log(f"{'─'*54}", "head")
            self._log(f"  Loaded: {path}", "head")
            self._log(f"  Shape : {self.df.shape}", "head")
            self._log(f"  Num   : {self.num_cols}", "head")
            self._log(f"  Cat   : {self.cat_cols}", "head")
            self._log(f"{'─'*54}\n", "head")
        except Exception as exc:
            self._status.config(text=f"Error: {exc}", fg=RED)
            self._log(f"[ERROR] {exc}", "err")

    # ── Button handlers (one per tool) ────────────────────────────────────────
    def _run_data_info(self):
        if not self._require_data(): return
        self._log("── Dataset Info ──────────────────────────────────────", "head")
        self._run(data_info, self.df)

    def _run_missing(self):
        if not self._require_data(): return
        self._log("── Missing Values ────────────────────────────────────", "head")
        self._run(missing_data, self.df)

    def _run_encoded(self):
        if not self._require_data(): return
        self._log("── Encoded Value Check ───────────────────────────────", "head")
        self._run(find_encoded_data, self.df)

    def _run_duplicates(self):
        if not self._require_data(): return
        self._log("── Duplicate Rows ────────────────────────────────────", "head")
        self._run(find_duplicates, self.df)

    def _run_cat_plots(self):
        if not self._require_data(): return
        self._log("── Categorical Plots ─────────────────────────────────", "head")
        self._run(countplot_cattegorical_columns, self.cat_cols, self.df)

    def _run_target_dist(self):
        if not self._require_data(): return
        target = self._target_var.get()
        self._log(f"── Target Distribution: {target} ──────────────────────", "head")
        self._run(target_col_distribution, target, self.df)

    def _run_num_dist(self):
        if not self._require_data(): return
        self._log("── Numeric Distributions ─────────────────────────────", "head")
        self._run(all_num_column_distribution, self.num_cols, self.df)

    def _run_boxplots(self):
        if not self._require_data(): return
        self._log("── Box Plots ─────────────────────────────────────────", "head")
        self._run(boxplot_visial, self.num_cols, self.df)

    def _run_corr_heatmap(self):
        if not self._require_data(): return
        self._log("── Correlation Heatmap ───────────────────────────────", "head")
        self._run(id_corrolated_col_features, self.num_cols, self.df)

    def _run_corr_target(self):
        if not self._require_data(): return
        target = self._target_var.get()
        self._log(f"── Correlation with Target: {target} ─────────────────", "head")
        self._run(corr_wth_trg_col, target, self.num_cols, self.df)

    def _clear_output(self):
        self._console.delete("1.0", tk.END)
        for w in self._plot_scroll.inner.winfo_children():
            w.destroy()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("All libraries imported successfully.")
    root = tk.Tk()
    app  = DataAnalyticsApp(root)
    root.mainloop()
