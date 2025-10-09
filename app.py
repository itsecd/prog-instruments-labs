import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from logging_conf import logger


# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Ultimate Dataset Visualizer", layout="wide")
logger.info("Streamlit app initialized - Ultimate Dataset Visualizer")

# -------------------------------
# Dark Mode Colors
# -------------------------------
bg_color = "#2C3E50"
text_color = "#EAEAEA"
card_color = "#3A506B"
hover_color = "#4B8BBE"

# -------------------------------
# CSS Styling
# -------------------------------
st.markdown(f"""
<style>
/* Main app background */
[data-testid="stAppViewContainer"], .stApp {{
    background-color: {bg_color};
    color: {text_color};
}}

/* Sidebar */
[data-testid="stSidebar"] {{
    background-color: {card_color};
    color: {text_color};
    border-radius: 10px;
    padding: 15px;
}}
[data-testid="stSidebar"] h2 {{color:#FF6F61; font-weight:bold;}}
[data-testid="stSidebar"] label {{color:{text_color}; font-size:16px;}}

/* Cards and Expanders */
.st-expander, .stMetric {{
    background-color: {card_color};
    color: {text_color};
    border-radius:10px;
    padding:10px;
    box-shadow:0 2px 6px rgba(0,0,0,0.1);
}}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Main Title
# -------------------------------
st.markdown(
    f"<h1 style='text-align:center; color:{text_color};'>Ultimate Dataset Visualizer</h1>",
    unsafe_allow_html=True)
st.markdown(
    f"<p style='text-align:center;color:{text_color};'>Professional multi-field data profiling & interactive visualization</p>",
    unsafe_allow_html=True)

# -------------------------------
# File Upload
# -------------------------------
st.sidebar.header("Upload CSV")
uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=["csv"])
st.sidebar.markdown("Navigate using tabs after upload.")

if uploaded_file:
    logger.info(
        f"File uploaded: {uploaded_file.name}, Size: {uploaded_file.size} bytes")

    try:
        df = pd.read_csv(uploaded_file)
        logger.info(
            f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        logger.error(f"Error loading CSV file: {str(e)}")
        st.error(f"Error loading file: {str(e)}")
        st.stop()


    # -------------------------------
    # Helper Functions
    # -------------------------------
    def friendly_dtype(dtype):
        if np.issubdtype(dtype, np.number):
            return "Integer" if np.issubdtype(dtype, np.integer) else "Float"
        elif np.issubdtype(dtype, np.datetime64):
            return "Datetime"
        else:
            return "Categorical"

    # Convert numeric columns to float
    numeric_conversion_count = 0
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
        numeric_conversion_count += 1
    logger.debug(
        f"Converted {numeric_conversion_count} numeric columns to float")

    # Convert datetime columns
    datetime_conversion_count = 0
    for col in df.select_dtypes(include=['datetime', 'datetime64']).columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        datetime_conversion_count += 1
    if datetime_conversion_count > 0:
        logger.debug(f"Converted {datetime_conversion_count} datetime columns")


    # Numeric Summary
    def numeric_summary(data, fields):
        logger.debug(f"Generating numeric summary for fields: {fields}")
        summary_list = []
        for col in fields:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers_count = data[(data[col] < Q1 - 1.5 * IQR) | (
                        data[col] > Q3 + 1.5 * IQR)].shape[0]
            desc = data[col].describe()
            summary_list.append({
                "Field": col,
                "Type": friendly_dtype(data[col].dtype),
                "Mean": round(desc["mean"], 2),
                "Std": round(desc["std"], 2),
                "Min": round(desc["min"], 2),
                "25%": round(Q1, 2),
                "50%": round(desc["50%"], 2),
                "75%": round(Q3, 2),
                "Max": round(desc["max"], 2),
                "Skewness": round(data[col].skew(), 2),
                "Kurtosis": round(data[col].kurtosis(), 2),
                "Outliers": outliers_count,
                "Missing": data[col].isnull().sum()
            })
        logger.debug(f"Numeric summary generated for {len(fields)} fields")
        return pd.DataFrame(summary_list)


    # Categorical Summary
    def categorical_summary(data, fields):
        logger.debug(f"Generating categorical summary for fields: {fields}")
        summary_list = []
        for col in fields:
            top_val = data[col].mode()[0] if not data[
                col].mode().empty else "N/A"
            top_freq = data[col].value_counts().iloc[0] if len(
                data[col].value_counts()) > 0 else 0
            summary_list.append({
                "Field": col,
                "Type": friendly_dtype(data[col].dtype),
                "Unique": data[col].nunique(),
                "Most Frequent": str(top_val),
                "Frequency": top_freq,
                "Missing": data[col].isnull().sum()
            })
        logger.debug(f"Categorical summary generated for {len(fields)} fields")
        return pd.DataFrame(summary_list)


    # -------------------------------
    # Tabs
    # -------------------------------
    overview_tab, numeric_tab, categorical_tab, datetime_tab, comparison_tab, export_tab = st.tabs(
        ["Overview", "Numeric Analysis", "Categorical Analysis",
         "Datetime Analysis", "Multi-Field Comparison", "Export"]
    )
    logger.debug("Application tabs initialized")

    # -------------------------------
    # Overview Tab
    # -------------------------------
    with overview_tab:
        logger.info("User navigated to Overview tab")
        st.header("Dataset Overview")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Entries", df.shape[0])
        col2.metric("Fields", df.shape[1])
        duplicates = df.duplicated().sum()
        col3.metric("Duplicates", duplicates)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(
            exclude=[np.number, 'datetime']).columns
        col4.metric("Numeric Fields", len(numeric_cols))
        col5.metric("Categorical Fields", len(categorical_cols))

        logger.info(
            f"Dataset stats - Rows: {df.shape[0]}, Cols: {df.shape[1]}, Duplicates: {duplicates}, "
            f"Numeric: {len(numeric_cols)}, Categorical: {len(categorical_cols)}")

        mem_usage = df.memory_usage(deep=True).sum() / (1024 ** 2)
        st.metric("Memory Usage (MB)", f"{mem_usage:.2f}")
        logger.debug(f"Memory usage: {mem_usage:.2f} MB")

        st.subheader("Missing Values Overview")
        missing_count = df.isnull().sum()
        missing_percent = (missing_count / len(df) * 100).round(2)
        total_missing = missing_count.sum()
        logger.info(f"Missing values analysis - Total missing: {total_missing}")
        st.dataframe(pd.DataFrame({"Missing Values": missing_count,
                                   "Percentage (%)": missing_percent}))

        st.subheader("Field Type Distribution")
        type_counts = df.dtypes.value_counts()
        logger.debug(f"Field type distribution: {type_counts.to_dict()}")
        fig = px.pie(values=type_counts.values,
                     names=type_counts.index.astype(str),
                     title="Field Type Distribution",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True, key="overview_type_dist")

        if df.isnull().sum().sum() > 0:
            st.subheader("Missing Values Heatmap")
            logger.debug("Generating missing values heatmap")
            fig_missing = px.imshow(df.isnull(),
                                    color_continuous_scale='Viridis',
                                    title="Missing Values Heatmap")
            st.plotly_chart(fig_missing, use_container_width=True,
                            key="overview_missing_heatmap")
        else:
            logger.debug("No missing values detected, skipping heatmap")

    # -------------------------------
    # Numeric Tab
    # -------------------------------
    with numeric_tab:
        logger.info("User navigated to Numeric Analysis tab")
        st.header("Numeric Fields Analysis")
        if len(numeric_cols) > 0:
            selected_num = st.selectbox("Select numeric field", numeric_cols,
                                        key="num_select")
            logger.debug(f"User selected numeric field: {selected_num}")

            summary_df = numeric_summary(df, [selected_num])
            st.subheader(f"Summary for {selected_num}")
            st.dataframe(summary_df)

            logger.debug(f"Generating histogram for {selected_num}")
            fig1 = px.histogram(df, x=selected_num, nbins=30,
                                title=f"Distribution: {selected_num}",
                                color_discrete_sequence=['#4B8BBE'])
            st.plotly_chart(fig1, use_container_width=True,
                            key=f"num_hist_{selected_num}")

            logger.debug(f"Generating boxplot for {selected_num}")
            fig2 = px.box(df, y=selected_num, title=f"Boxplot: {selected_num}",
                          color_discrete_sequence=['#FF6F61'])
            st.plotly_chart(fig2, use_container_width=True,
                            key=f"num_box_{selected_num}")

            st.subheader("Correlation with other numeric fields")
            logger.debug("Generating correlation matrix")
            corr_matrix = df[numeric_cols].corr()
            fig_corr = px.imshow(corr_matrix, text_auto=True,
                                 color_continuous_scale='RdBu_r',
                                 title="Correlation Matrix")
            st.plotly_chart(fig_corr, use_container_width=True, key="num_corr")
        else:
            logger.warning("No numeric fields detected in dataset")
            st.info("No numeric fields detected.")

    # -------------------------------
    # Categorical Tab
    # -------------------------------
    with categorical_tab:
        logger.info("User navigated to Categorical Analysis tab")
        st.header("Categorical Fields Analysis")
        if len(categorical_cols) > 0:
            selected_cat = st.selectbox("Select categorical field",
                                        categorical_cols, key="cat_select")
            logger.debug(f"User selected categorical field: {selected_cat}")

            summary_df = categorical_summary(df, [selected_cat])
            st.subheader(f"Summary for {selected_cat}")
            st.dataframe(summary_df)

            st.subheader("Top Values")
            counts = df[selected_cat].value_counts()
            counts.index = counts.index.astype(str)
            logger.debug(
                f"Top values for {selected_cat}: {len(counts)} unique values")
            fig_bar = px.bar(counts, title=f"Top Values: {selected_cat}")
            st.plotly_chart(fig_bar, use_container_width=True,
                            key=f"cat_bar_{selected_cat}")
        else:
            logger.warning("No categorical fields detected in dataset")
            st.info("No categorical fields detected.")

    # -------------------------------
    # Datetime Tab
    # -------------------------------
    with datetime_tab:
        logger.info("User navigated to Datetime Analysis tab")
        st.header("Datetime Fields Analysis")
        datetime_cols = df.select_dtypes(
            include=['datetime', 'datetime64']).columns
        if len(datetime_cols) > 0:
            selected_dt = st.selectbox("Select datetime field", datetime_cols,
                                       key="dt_select")
            logger.debug(f"User selected datetime field: {selected_dt}")

            df[selected_dt] = pd.to_datetime(df[selected_dt], errors='coerce')
            st.subheader(f"Analysis for {selected_dt}")
            date_range = f"{df[selected_dt].min()} to {df[selected_dt].max()}"
            missing_dt = df[selected_dt].isnull().sum()
            st.write(f"- Range: {date_range}")
            st.write(f"- Missing: {missing_dt}")
            logger.info(
                f"Datetime analysis - Field: {selected_dt}, Range: {date_range}, Missing: {missing_dt}")
            counts_dt = df[selected_dt].value_counts().sort_index()
            counts_dt.index = counts_dt.index.astype(str)
            logger.debug(
                f"Generating timeline for {selected_dt} with {len(counts_dt)} data points")
            fig_dt = px.line(counts_dt,
                             title=f"Value Counts Over Time: {selected_dt}")
            st.plotly_chart(fig_dt, use_container_width=True,
                            key=f"dt_line_{selected_dt}")
        else:
            logger.info("No datetime fields detected in dataset")
            st.info("No datetime fields detected.")

    # -------------------------------
    # Multi-Field Comparison Tab
    # -------------------------------
    with comparison_tab:
        logger.info("User navigated to Multi-Field Comparison tab")
        st.header("Multi-Field Comparison")
        multi_numeric = st.multiselect("Select numeric fields", numeric_cols,
                                       key="multi_num")
        multi_categorical = st.multiselect("Select categorical fields",
                                           categorical_cols, key="multi_cat")
        group_by_cat = st.selectbox("Optional: Group by categorical field",
                                    options=[None] + list(categorical_cols),
                                    key="group_by")
        logger.debug(
            f"Multi-field selection - Numeric: {multi_numeric}, Categorical: {multi_categorical}, Group by: {group_by_cat}")
        if multi_numeric or multi_categorical:
            with st.expander("Side-by-Side Summary"):
                if multi_numeric:
                    logger.debug(
                        f"Generating side-by-side numeric summary for {len(multi_numeric)} fields")
                    num_comp_df = numeric_summary(df, multi_numeric)
                    st.dataframe(num_comp_df)
                if multi_categorical:
                    logger.debug(
                        f"Generating side-by-side categorical summary for {len(multi_categorical)} fields")
                    cat_comp_df = categorical_summary(df, multi_categorical)
                    st.dataframe(cat_comp_df)

            with st.expander("Comparison Plots"):
                for idx, col in enumerate(multi_numeric):
                    logger.debug(
                        f"Generating comparison plots for numeric field: {col}")
                    fig = px.histogram(df, x=col, nbins=30,
                                       title=f"Distribution: {col}",
                                       color_discrete_sequence=['#4B8BBE'])
                    st.plotly_chart(fig, use_container_width=True,
                                    key=f"comp_hist_{col}_{idx}")
                    fig_box = px.box(df, y=col, title=f"Boxplot: {col}",
                                     color_discrete_sequence=['#FF6F61'])
                    st.plotly_chart(fig_box, use_container_width=True,
                                    key=f"comp_box_{col}_{idx}")
                for idx, col in enumerate(multi_categorical):
                    logger.debug(
                        f"Generating comparison plots for categorical field: {col}")
                    counts = df[col].value_counts()
                    counts.index = counts.index.astype(str)
                    fig_cat = px.bar(counts, title=f"Top Values: {col}")
                    st.plotly_chart(fig_cat, use_container_width=True,
                                    key=f"comp_cat_{col}_{idx}")

            if group_by_cat and multi_numeric:
                logger.info(
                    f"Generating grouped analysis by {group_by_cat} for {len(multi_numeric)} numeric fields")
                st.subheader(f"Grouped Analysis by {group_by_cat}")
                grouped = df.groupby(group_by_cat)[multi_numeric].mean()
                st.dataframe(grouped)
                st.bar_chart(grouped)
        else:
            logger.debug("No fields selected for multi-field comparison")

    # -------------------------------
    # Export Tab
    # -------------------------------
    with export_tab:
        logger.info("User navigated to Export tab")
        st.header("Export Professional Summary")
        export_num = numeric_summary(df, numeric_cols) if len(
            numeric_cols) > 0 else pd.DataFrame()
        export_cat = categorical_summary(df, categorical_cols) if len(
            categorical_cols) > 0 else pd.DataFrame()
        export_df = pd.concat([export_num, export_cat], ignore_index=True,
                              sort=False)

        csv = export_df.to_csv(index=False).encode('utf-8')
        logger.info(f"Export data prepared - {len(export_df)} summary rows")

        st.download_button("Download Professional CSV", data=csv,
                           file_name="dataset_summary_professional.csv",
                           mime="text/csv")
        logger.debug("Export download button displayed")

else:
    logger.debug("No file uploaded yet - displaying upload prompt")
    st.info("Please upload a CSV file to start exploring your dataset.")

logger.info("Application session completed")
