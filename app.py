# --------------------------------------------------
# 📊 DataViz Studio: An Interactive Data Analytics Dashboard
# --------------------------------------------------

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# --- Streamlit setup ---
st.set_page_config(page_title="DataViz Studio", layout="wide")
st.title("📊 DataViz Studio: An Interactive Data Analytics Dashboard")

# --- File upload ---
st.sidebar.header("📂 Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a data file", type=["csv", "json", "xlsx"])

if uploaded_file is not None:
    # Load dataset dynamically
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".json"):
        df = pd.read_json(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("❌ Unsupported file type.")
        st.stop()

    st.success(f"✅ Successfully loaded **{uploaded_file.name}**")
    st.dataframe(df.head())

    if "graphs" not in st.session_state:
        st.session_state["graphs"] = []

    # --- EDA Section ---
    with st.expander("📈 Explore Dataset"):
        st.write("### Dataset Summary")
        st.dataframe(df.describe().T)

        st.write("### Missing Values")
        st.dataframe(df.isnull().sum())

        st.write("### Column Data Types")
        st.dataframe(df.dtypes)

    # --- Correlation Heatmap ---
        st.write("### 🔥 Correlation Heatmap")
        numeric_df = df.select_dtypes(include=["number"])

        if numeric_df.shape[1] > 1:
            corr = numeric_df.corr()
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax, fmt=".2f", linewidths=0.5)
            ax.set_title("Correlation Matrix", fontsize=12, fontweight="bold")
            st.pyplot(fig)
        else:
            st.info("ℹ️ Not enough numerical columns for correlation analysis.")


    # --- Visualization Section ---
    st.sidebar.header("🎨 Create Visualizations")
    x_axis = st.sidebar.selectbox("Select X-axis", df.columns)
    y_axis = st.sidebar.selectbox("Select Y-axis", df.columns)
    graph_type = st.sidebar.selectbox(
        "Select Graph Type",
        ["Line", "Bar", "Scatter", "Histogram", "Boxplot", "Pie"]
    )
    graph_size = st.sidebar.slider("Graph Size (Adjust Zoom)", 4, 16, 8)

    # 🎨 Color Customization
    main_color = st.sidebar.color_picker("Main Color", "#1f77b4")
    edge_color = st.sidebar.color_picker("Edge/Outline Color", "#000000")

    # --- Graph Creation ---
    if st.sidebar.button("Generate Graph"):
        fig, ax = plt.subplots(figsize=(graph_size, graph_size / 1.5))

        try:
            if graph_type == "Line":
                ax.plot(df[x_axis], df[y_axis], color=main_color, linewidth=2)
            elif graph_type == "Bar":
                ax.bar(df[x_axis], df[y_axis], color=main_color, edgecolor=edge_color)
            elif graph_type == "Scatter":
                ax.scatter(df[x_axis], df[y_axis], color=main_color, edgecolors=edge_color, s=60)
            elif graph_type == "Histogram":
                ax.hist(df[x_axis], bins=20, color=main_color, edgecolor=edge_color, alpha=0.7)
            elif graph_type == "Boxplot":
                ax.boxplot(df[y_axis].dropna(), boxprops=dict(color=main_color), medianprops=dict(color=edge_color))
                ax.set_xticklabels([y_axis])
            elif graph_type == "Pie":
                if df[x_axis].dtype == 'object' or df[x_axis].nunique() < 15:
                    data = df[x_axis].value_counts()
                    ax.pie(data.values, labels=data.index, autopct='%1.1f%%',
                           startangle=90, colors=plt.cm.tab10.colors)
                    ax.set_title(f"Pie Chart of {x_axis}")
                    ax.axis("equal")
                else:
                    st.warning("⚠️ Pie chart is best for categorical columns or few unique values.")
                    st.stop()

            ax.set_xlabel(x_axis)
            if graph_type != "Pie":
                ax.set_ylabel(y_axis)
            ax.set_title(f"{graph_type} of {y_axis} vs {x_axis}" if graph_type != "Pie" else f"{graph_type} of {x_axis}")

            st.pyplot(fig)
            st.session_state["graphs"].append((fig, f"{graph_type} of {y_axis} vs {x_axis}"))

        except Exception as e:
            st.error(f"Error: {e}")

    # --- Combined Dashboard View ---
    if st.session_state["graphs"]:
        st.subheader("🧩 Combined Dashboard View")

        num_graphs = len(st.session_state["graphs"])
        cols = 2 if num_graphs >= 4 else 1
        rows = (num_graphs + cols - 1) // cols

        fig, axs = plt.subplots(rows, cols, figsize=(10, 5 * rows))
        axs = axs.flatten() if num_graphs > 1 else [axs]

        for i, (graph_data, ax) in enumerate(zip(st.session_state["graphs"], axs)):
            graph, _ = graph_data
            graph.canvas.draw()
            img = np.asarray(graph.canvas.buffer_rgba())
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"Graph {i+1}", fontsize=10)

        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()
        st.pyplot(fig)

        # --- Clean PDF Report (Stats + Correlation + Dashboard) ---
        pdf_buffer = BytesIO()
        with PdfPages(pdf_buffer) as pdf:
            # Page 1: Overview and Stats
            stats_fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis("off")

            # Title
            ax.text(0.5, 0.95, "📊 DataViz Studio Report", fontsize=18, fontweight="bold", ha="center")

            # File info
            ax.text(0.05, 0.90, f"File Name: {uploaded_file.name}", fontsize=11)
            ax.text(0.05, 0.87, f"Rows: {df.shape[0]}    Columns: {df.shape[1]}", fontsize=11)

            # Column Data Types Table
            ax.text(0.05, 0.82, "Column Data Types:", fontsize=12, fontweight="bold")
            col_dtypes = pd.DataFrame(df.dtypes.astype(str), columns=["Data Type"])
            ax.table(cellText=col_dtypes.values,
                     colLabels=col_dtypes.columns,
                     rowLabels=col_dtypes.index,
                     loc="upper left",
                     colWidths=[0.3],
                     bbox=[0.05, 0.45, 0.4, 0.35])

            # Statistical Summary Table
            ax.text(0.05, 0.40, "Statistical Summary:", fontsize=12, fontweight="bold")
            summary = df.describe().round(2).reset_index()
            ax.table(cellText=summary.values,
                     colLabels=summary.columns,
                     loc="upper left",
                     colWidths=[0.12]*len(summary.columns),
                     bbox=[0.05, 0.05, 0.9, 0.35])

            pdf.savefig(stats_fig, bbox_inches="tight")
            plt.close(stats_fig)

            # Page 2: Correlation Heatmap (if available)
            numeric_df = df.select_dtypes(include=["number"])
            if numeric_df.shape[1] > 1:
                corr_fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
                ax.set_title("Correlation Heatmap", fontsize=14, fontweight="bold")
                pdf.savefig(corr_fig, bbox_inches="tight")
                plt.close(corr_fig)

            # Page 3: Combined Dashboard
            pdf.savefig(fig, bbox_inches="tight")

        pdf_buffer.seek(0)

        st.download_button(
            label="📄 Download Final Report",
            data=pdf_buffer,
            file_name="DataVizStudio_Report.pdf",
            mime="application/pdf"
        )

    else:
        st.info("👈 Upload a CSV, JSON, or Excel file to get started.")
