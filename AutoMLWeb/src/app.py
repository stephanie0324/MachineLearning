import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# Set up Streamlit page
st.set_page_config(page_title="AutoML CSV Trainer", layout="wide")
st.title("🧠 AutoML CSV Trainer")
st.markdown("""
Upload your CSV, select the label column (or none for clustering), 
and experience automatic EDA, model training, feature selection, and score comparison.  
*Optimized for Apple Silicon (M1/M2/M3) environments.*
""")

# --- Task detector ---
def detect_task_type(df, target_col=None):
    if target_col is None or target_col == "No target (clustering)":
        return "clustering"
    y = df[target_col]
    if y.dtype.kind in "biu":  # integers
        if y.nunique() <= 20:
            return "classification"
        else:
            return "regression"
    elif y.dtype.kind in "f":  # floats
        return "regression"
    else:  # strings / categories
        return "classification"

# File upload
uploaded_file = st.file_uploader("📄 Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Add clustering option
    target_col = st.selectbox(
        "Select your label column (or choose clustering)",
        ["No target (clustering)"] + df.columns.tolist()
    )
    
    # 👇 Show detected task right after user picks target
    task_type_preview = detect_task_type(df, target_col)
    if target_col == "No target (clustering)":
        st.info("🔍 Task detected: **Clustering** (no target selected)")
    elif task_type_preview == "classification":
        st.info("🔍 Task detected: **Classification** (target is categorical / discrete)")
    elif task_type_preview == "regression":
        st.info("🔍 Task detected: **Regression** (target is numeric / continuous)")

    # Main workflow tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1️⃣ EDA", "2️⃣ Model Training", "3️⃣ Results", "4️⃣ Feature Selection & Retrain", "5️⃣ Score Comparison"
    ])

    # --- 1. EDA ---
    with tab1:
        st.subheader("Data Preview")
        st.dataframe(df.head())
        st.write("Basic Statistics")
        st.dataframe(df.describe())

        # Dynamically select columns suitable for pie charts
        pie_chart_cols = []
        for col in df.columns:
            unique_count = df[col].nunique(dropna=True)
            if pd.api.types.is_integer_dtype(df[col]):
                if unique_count <= 20:
                    pie_chart_cols.append((col, 'category'))
            elif pd.api.types.is_float_dtype(df[col]):
                if unique_count > 20:
                    pie_chart_cols.append((col, 'bucket'))
            else:
                if unique_count <= 10:
                    pie_chart_cols.append((col, 'category'))

        col_objs = st.columns(len(pie_chart_cols))
        for i, (col, col_type) in enumerate(pie_chart_cols):
            with col_objs[i]:
                st.markdown(f"**{col} Distribution**")
                try:
                    if col_type == 'bucket':
                        buckets = pd.cut(df[col].dropna(), bins=10)
                        pie_data = buckets.value_counts().sort_index()
                        pie_data.index = [f"{interval.left:.0f}~{interval.right:.0f}" for interval in pie_data.index]
                    else:
                        pie_data = df[col].dropna().value_counts()
                    fig = pie_data.plot.pie(
                        labels=pie_data.index, 
                        autopct='%1.1f%%', 
                        figsize=(4, 4), 
                        legend=False
                    ).get_figure()
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Could not plot pie chart for {col}: {e}")

    # --- 2. Model Training ---
    with tab2:
        if st.button("🚀 Run AutoML", key="run1"):
            task_type = detect_task_type(df, target_col)
            st.session_state['task_type'] = task_type
            st.session_state['target_col'] = None if target_col == "No target (clustering)" else target_col

            if task_type == "classification":
                from pycaret.classification import setup, compare_models, pull
                setup(data=df, target=st.session_state['target_col'], html=False, session_id=42)
                best_model = compare_models()
                results = pull()

            elif task_type == "regression":
                from pycaret.regression import setup, compare_models, pull
                setup(data=df, target=st.session_state['target_col'], html=False, session_id=42)
                best_model = compare_models()
                results = pull()

            elif task_type == "clustering":
                from pycaret.clustering import setup, create_model, pull
                setup(data=df, html=False, session_id=42)
                best_model = create_model("kmeans")
                results = pull()

            st.session_state['results1'] = results
            st.session_state['best_model1'] = best_model
            st.success(f"AutoML complete! Detected task: **{task_type}**. Switch to the Results tab.")

    # --- 3. Results ---
    with tab3:
        if 'results1' in st.session_state:
            results = st.session_state['results1']
            st.subheader("🏆 Best Model")
            st.code(str(st.session_state['best_model1']))
            st.subheader("Detailed Model Scores / Summary")
            st.dataframe(results)
        else:
            st.info("Please run AutoML in the Model Training tab first.")

    # --- 4. Feature Selection & Retraining ---
    with tab4:
        if 'target_col' in st.session_state and st.session_state['target_col'] is not None:
            if st.button("✨ Feature Selection & Retrain", key="run2"):
                if st.session_state['task_type'] == "classification":
                    from pycaret.classification import setup, compare_models, pull
                else:
                    from pycaret.regression import setup, compare_models, pull

                s = setup(
                    data=df,
                    target=st.session_state['target_col'],
                    html=False,
                    session_id=42,
                    feature_selection=True,
                    feature_selection_estimator="rf"
                )
                selected_features = s.X.columns.tolist()
                st.session_state['selected_features'] = selected_features
                df_fs = df[selected_features + [st.session_state['target_col']]]
                setup(data=df_fs, target=st.session_state['target_col'], html=False, session_id=42)
                best_model_fs = compare_models()
                results_fs = pull()
                st.session_state['results2'] = results_fs
                st.session_state['best_model2'] = best_model_fs
                st.success("Feature selection + retraining complete! Check Score Comparison tab.")
        else:
            st.info("Feature selection only applies to classification/regression tasks.")

    # --- 5. Score Comparison ---
    with tab5:
        if 'results1' in st.session_state and 'results2' in st.session_state:
            score_col = "Accuracy" if "Accuracy" in st.session_state['results1'].columns else st.session_state['results1'].columns[-1]
            score1 = st.session_state['results1'].iloc[0][score_col]
            score2 = st.session_state['results2'].iloc[0][score_col]
            st.subheader("Best Model Score Comparison")
            st.metric("Original Score", f"{score1:.3f}")
            st.metric("After Feature Selection", f"{score2:.3f}")
            st.line_chart(pd.DataFrame({
                "Original": [score1],
                "Feature Selection": [score2]
            }))
            if 'selected_features' in st.session_state:
                st.write("Selected Features:", st.session_state['selected_features'])
        else:
            st.info("Run classification/regression training and retrain before comparing.")
