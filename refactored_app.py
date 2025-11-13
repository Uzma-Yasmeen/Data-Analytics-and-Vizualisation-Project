# Refactored DataViz Studio - Robust, production-ready version
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="DataViz Studio (Refactored)", layout="wide")
st.title("📊 DataViz Studio — Robust Analytics Dashboard")

# Helpers
def safe_str(v):
    try:
        if pd.isna(v):
            return ""
        return str(v)
    except Exception:
        return str(v)

def to_numeric_series(s):
    """Try to convert a pandas Series to numeric, coerce errors to NaN"""
    try:
        return pd.to_numeric(s, errors='coerce')
    except Exception:
        return s

def prepare_xy(df, x_col, y_col, graph_type):
    """Return x_vals, y_vals prepared for plotting based on types and graph type"""
    x = df[x_col]
    y = df[y_col] if y_col in df.columns else None

    # For plotting, convert categorical x to string
    if x.dtype.name == 'category' or x.dtype == object:
        x_vals = x.astype(str).fillna('')
    else:
        # numeric or datetime: keep as-is but convert to numeric if needed for histogram
        x_vals = x
    # y conversion for numeric plots
    if y is not None:
        if graph_type in ['Line', 'Bar', 'Scatter', 'Boxplot']:
            y_vals = to_numeric_series(y).fillna(0)
        else:
            y_vals = y
    else:
        y_vals = None
    return x_vals, y_vals

# File upload
st.sidebar.header("📂 Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a data file", type=["csv", "json", "xlsx"])

if uploaded_file is None:
    st.info("👈 Start by uploading a CSV, JSON, or Excel file.")
    st.stop()

# Load dataset robustly
try:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".json"):
        df = pd.read_json(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("❌ Unsupported file type.")
        st.stop()
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

st.success(f"✅ Successfully loaded **{uploaded_file.name}** ({df.shape[0]} rows × {df.shape[1]} cols)")
st.dataframe(df.head())

# Normalize column names to strings for UI
df.columns = [safe_str(c) for c in df.columns]

# Initialize session state store for graph specs (not raw figure objects)
if "graph_specs" not in st.session_state:
    st.session_state["graph_specs"] = []

# EDA
with st.expander("📈 Explore Dataset"):
    st.write("### Dataset Summary (Numeric columns)")
    try:
        st.dataframe(df.describe().T.style.format("{:.2f}"))
    except Exception:
        st.dataframe(df.describe().T)

    st.write("### Missing Values (per column)")
    st.dataframe(df.isnull().sum())

    st.write("### Column Data Types")
    dtypes = pd.DataFrame(df.dtypes.astype(str), columns=["Data Type"])
    st.dataframe(dtypes)

    # Correlation heatmap
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] > 1:
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax, fmt=".2f", linewidths=0.5)
        ax.set_title("Correlation Matrix", fontsize=12, fontweight="bold")
        st.pyplot(fig)
    else:
        st.info("ℹ️ Not enough numerical columns for correlation analysis.")

# Visualization options
st.sidebar.header("🎨 Create Visualizations")
cols = df.columns.tolist()
x_axis = st.sidebar.selectbox("Select X-axis", cols, index=0)
# allow user to optionally select Y; some chart types use only X
y_axis = st.sidebar.selectbox("Select Y-axis (if applicable)", [""] + cols, index=0)
graph_type = st.sidebar.selectbox("Select Graph Type", ["Line", "Bar", "Scatter", "Histogram", "Boxplot", "Pie"])
graph_size = st.sidebar.slider("Graph Size (Adjust zoom)", 4, 16, 8)
main_color = st.sidebar.color_picker("Main Color", "#1f77b4")
edge_color = st.sidebar.color_picker("Edge/Outline Color", "#000000")

# Generate graph button
if st.sidebar.button("Generate Graph"):
    try:
        # Prepare data for plotting
        if graph_type == "Pie":
            # Pie uses categorical counts of x_axis
            data = df[x_axis].astype(str).fillna("(missing)").value_counts()
            labels = data.index.tolist()
            sizes = data.values
            fig, ax = plt.subplots(figsize=(graph_size, graph_size/1.5))
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.tab10.colors)
            ax.set_title(f"Pie Chart of {x_axis}")
            ax.axis('equal')
        elif graph_type == "Histogram":
            x_vals = to_numeric_series(df[x_axis]).dropna()
            if x_vals.empty:
                st.warning("No numeric data found in selected column for histogram.")
                st.stop()
            fig, ax = plt.subplots(figsize=(graph_size, graph_size/1.5))
            ax.hist(x_vals, bins=20, color=main_color, edgecolor=edge_color, alpha=0.8)
            ax.set_title(f"Histogram of {x_axis}")
            ax.set_xlabel(x_axis)
            ax.set_ylabel("Count")
        else:
            # For other charts we need x and y
            if y_axis == "":
                st.warning("Please select a Y-axis column for this chart type.")
                st.stop()
            x_vals, y_vals = prepare_xy(df, x_axis, y_axis, graph_type)
            fig, ax = plt.subplots(figsize=(graph_size, graph_size/1.5))
            if graph_type == "Line":
                ax.plot(x_vals, y_vals, color=main_color, linewidth=2)
            elif graph_type == "Bar":
                # aggregate if x is categorical
                if x_vals.dtype == object or x_vals.nunique() < 30:
                    agg = pd.DataFrame({y_axis: y_vals, x_axis: x_vals}).groupby(x_axis).mean()[y_axis]
                    ax.bar(agg.index.astype(str), agg.values, color=main_color, edgecolor=edge_color)
                else:
                    ax.bar(x_vals.astype(str), y_vals, color=main_color, edgecolor=edge_color)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            elif graph_type == "Scatter":
                ax.scatter(x_vals, y_vals, color=main_color, edgecolors=edge_color, s=60)
            elif graph_type == "Boxplot":
                ax.boxplot(y_vals.dropna(), boxprops=dict(color=main_color), medianprops=dict(color=edge_color))
                ax.set_xticklabels([y_axis])

            ax.set_xlabel(x_axis)
            ax.set_ylabel(y_axis)
            ax.set_title(f"{graph_type} of {y_axis} vs {x_axis}")

        st.pyplot(fig)

        # Save graph spec (parameters) for combined dashboard & PDF
        spec = {"type": graph_type, "x": x_axis, "y": y_axis, "size": graph_size,
                "main_color": main_color, "edge_color": edge_color, "title": fig._suptitle.get_text() if fig._suptitle else ""}
        st.session_state["graph_specs"].append(spec)

    except Exception as e:
        st.error(f"Error while generating graph: {e}")
        logging.exception(e)

# Combined Dashboard view: recreate figures from specs
if st.session_state["graph_specs"]:
    st.subheader("🧩 Combined Dashboard View")
    specs = st.session_state["graph_specs"]
    n = len(specs)
    cols = 2 if n >= 2 else 1
    rows = (n + cols - 1) // cols
    fig_comb, axs = plt.subplots(rows, cols, figsize=(10, 4 * rows))
    axs = np.array(axs).reshape(-1) if isinstance(axs, np.ndarray) else np.array([axs])

    for i, spec in enumerate(specs):
        ax = axs[i]
        try:
            gtype = spec["type"]
            gx = spec["x"]; gy = spec["y"]
            gsize = spec["size"]
            gmain = spec["main_color"]; gedge = spec["edge_color"]

            if gtype == "Pie":
                data = df[gx].astype(str).fillna("(missing)").value_counts()
                ax.pie(data.values, labels=data.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.tab10.colors)
                ax.set_title(f"Pie Chart of {gx}")
                ax.axis('equal')
            elif gtype == "Histogram":
                xvals = to_numeric_series(df[gx]).dropna()
                ax.hist(xvals, bins=20, color=gmain, edgecolor=gedge, alpha=0.8)
                ax.set_title(f"Histogram of {gx}")
            else:
                if gy == "":
                    ax.text(0.5, 0.5, "No Y selected", ha='center')
                else:
                    xvals, yvals = prepare_xy(df, gx, gy, gtype)
                    if gtype == "Line":
                        ax.plot(xvals, yvals, color=gmain, linewidth=2)
                    elif gtype == "Bar":
                        if xvals.dtype == object or xvals.nunique() < 30:
                            agg = pd.DataFrame({gy: yvals, gx: xvals}).groupby(gx).mean()[gy]
                            ax.bar(agg.index.astype(str), agg.values, color=gmain, edgecolor=gedge)
                        else:
                            ax.bar(xvals.astype(str), yvals, color=gmain, edgecolor=gedge)
                        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                    elif gtype == "Scatter":
                        ax.scatter(xvals, yvals, color=gmain, edgecolors=gedge, s=40)
                    elif gtype == "Boxplot":
                        ax.boxplot(yvals.dropna(), boxprops=dict(color=gmain), medianprops=dict(color=gedge))
                        ax.set_xticklabels([gy])
                    ax.set_xlabel(gx)
                    ax.set_ylabel(gy)
            ax.set_title(spec.get("title") or f"{gtype} of {spec.get('y') or spec.get('x')}")

        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {safe_str(e)}", ha='center')
            logging.exception(e)

    # hide extra axes
    for j in range(i+1, len(axs)):
        try:
            fig_comb.delaxes(axs[j])
        except Exception:
            pass

    plt.tight_layout()
    st.pyplot(fig_comb)

    # PDF export: regenerate each figure and save
    pdf_buffer = BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        # Page 1: Overview
        stats_fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.text(0.5, 0.95, "📊 DataViz Studio - Report", fontsize=18, fontweight="bold", ha="center")
        ax.text(0.05, 0.90, f"File: {safe_str(uploaded_file.name)}", fontsize=11)
        ax.text(0.05, 0.87, f"Rows: {df.shape[0]}    Columns: {df.shape[1]}", fontsize=11)
        # Column dtypes table
        col_dtypes = pd.DataFrame(df.dtypes.astype(str), columns=["Data Type"])
        table = ax.table(cellText=col_dtypes.values, colLabels=col_dtypes.columns, rowLabels=col_dtypes.index,
                         loc="upper left", bbox=[0.05, 0.55, 0.9, 0.3])
        pdf.savefig(stats_fig, bbox_inches="tight")
        plt.close(stats_fig)

        # Following pages: recreate each saved spec as figure and save
        for spec in specs:
            figp, axp = plt.subplots(figsize=(8, 6))
            try:
                if spec["type"] == "Pie":
                    data = df[spec["x"]].astype(str).fillna("(missing)").value_counts()
                    axp.pie(data.values, labels=data.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.tab10.colors)
                elif spec["type"] == "Histogram":
                    xvals = to_numeric_series(df[spec["x"]]).dropna()
                    axp.hist(xvals, bins=20, color=spec["main_color"], edgecolor=spec["edge_color"], alpha=0.8)
                else:
                    if spec["y"] == "":
                        axp.text(0.5, 0.5, "No Y selected", ha='center')
                    else:
                        xv, yv = prepare_xy(df, spec["x"], spec["y"], spec["type"])
                        if spec["type"] == "Line":
                            axp.plot(xv, yv, color=spec["main_color"])
                        elif spec["type"] == "Bar":
                            if xv.dtype == object or xv.nunique() < 30:
                                agg = pd.DataFrame({spec["y"]: yv, spec["x"]: xv}).groupby(spec["x"]).mean()[spec["y"]]
                                axp.bar(agg.index.astype(str), agg.values, color=spec["main_color"])
                            else:
                                axp.bar(xv.astype(str), yv, color=spec["main_color"])
                        elif spec["type"] == "Scatter":
                            axp.scatter(xv, yv, color=spec["main_color"], edgecolors=spec["edge_color"], s=40)
                        elif spec["type"] == "Boxplot":
                            axp.boxplot(yv.dropna(), boxprops=dict(color=spec["main_color"]), medianprops=dict(color=spec["edge_color"]))
                axp.set_title(spec.get("title") or f"{spec['type']} of {spec.get('y') or spec.get('x')}")
                pdf.savefig(figp, bbox_inches="tight")
                plt.close(figp)
            except Exception as e:
                logging.exception(e)
                axp.text(0.5, 0.5, f"Could not render: {safe_str(e)}", ha='center')
                pdf.savefig(figp, bbox_inches="tight")
                plt.close(figp)

    pdf_buffer.seek(0)
    st.download_button(label="📄 Download Final Report (PDF)", data=pdf_buffer, file_name="DataVizStudio_Report.pdf", mime="application/pdf")

else:
    st.info("👈 Generate at least one graph to build a combined dashboard and report.")
