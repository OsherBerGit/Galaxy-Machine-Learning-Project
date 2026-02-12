import os
import sys

# --- 1. Path Setup ---
project_root = os.path.dirname(__file__)
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import streamlit as st
import pandas as pd
import pickle
from PIL import Image

# --- Load Custom Modules ---
try:
    from feature_extractor_single import extract_features_from_single_image
except ImportError as e:
    st.error(f"Import error: {e}")

# --- Page Config ---
st.set_page_config(
    page_title="Galaxy Classifier AI",
    page_icon="🌌",
    layout="wide"
)

# --- Dynamic Data Loading Functions ---
# This function reads the actual results file to display real numbers
def get_real_model_accuracy():
    csv_path = os.path.join(project_root, 'data', 'grid_search_summary.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Find the highest accuracy for our model (Decision Tree with most iterations)
        # Assuming we use 200 iterations as defined
        row = df[(df['Model'] == 'Decision Tree') & (df['Total_Iters'] == 200)]
        if not row.empty:
            return row.iloc[0]['Final_Accuracy']
    return 0.0 # Default if file doesn't exist

# Function to find the most important feature from the logs
def get_top_feature_from_logs():
    log_path = os.path.join(project_root, 'data', 'grid_search_detailed_logs.csv')
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        # Extract feature name from the rule (e.g., "Is std_blue <= ...")
        df['Feature_Name'] = df['Rule'].apply(lambda x: x.split()[1] if isinstance(x, str) and "Is " in x else None)
        top_feature = df['Feature_Name'].mode()[0] # Most common feature
        return top_feature
    return "Unknown"

# --- Load Model (Cached) ---
@st.cache_resource
def load_model():
    try:
        import galaxy_adaboost
        sys.modules['galaxy_adaboost'] = galaxy_adaboost
        
        model_path = os.path.join(project_root, 'models', 'final_galaxy_model.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- Helper for Plots ---
def show_plot_if_exists(filename, caption):
    full_path = os.path.join(project_root, 'plots', filename)
    if os.path.exists(full_path):
        image = Image.open(full_path)
        st.image(image, caption=caption, use_container_width=True)
    else:
        st.warning(f"Plot not found: {filename}")

# ==========================================
# MAIN APP LOGIC
# ==========================================

# Read the real data before anything else
real_acc = get_real_model_accuracy()
top_feat = get_top_feature_from_logs()

st.sidebar.title("Navigation 🧭")
app_mode = st.sidebar.radio("Choose Mode:", [
    "🚀 Live Prediction", 
    "📊 Research: Feature Analysis", 
    "🧠 Research: Boosting Internals", 
    "⚖️ Research: Bias-Variance Analysis",
    "🏆 Research: Model Comparison"
])

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Current Model Status:**")
st.sidebar.markdown(f"✅ Trained on: **Decision Tree Base**")
st.sidebar.markdown(f"✅ Accuracy: **{real_acc*100:.2f}%** (Loaded from CSV)")
st.sidebar.markdown(f"✅ Top Feature: **{top_feat}**")

# --- MODE 1: LIVE PREDICTION ---
if app_mode == "🚀 Live Prediction":
    model = load_model()

    st.title("🌌 Galaxy Type Detector")
    st.markdown(f"""
    Welcome. This system uses a manual implementation of **AdaBoost**.
    Based on our training data, this model achieves an accuracy of **{real_acc*100:.2f}%**.
    """)
    st.markdown("---")

    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        with col1:
            st.image(image, caption='Uploaded Galaxy', use_container_width=True)

        temp_path = os.path.join(project_root, "temp_image.jpg")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with col2:
            st.markdown("### Analysis")
            if st.button("Analyze Galaxy 🚀", type="primary"):
                if model is None:
                    st.error("Model file missing.")
                else:
                    with st.spinner('Processing...'):
                        try:
                            # A. Extract features
                            features_array = extract_features_from_single_image(temp_path)
                            
                            # B. Predict
                            prediction = model.predict(features_array)[0]
                            
                            # C. Result
                            if prediction == 1:
                                st.success("Result: 🌀 SPIRAL GALAXY")
                            else:
                                st.warning("Result: 🥚 ELLIPTICAL GALAXY")

                            # D. Data Table
                            st.markdown("#### Extracted Features:")
                            feature_names = [
                                'Mean Red', 'Mean Green', 'Mean Blue',
                                'Std Red', 'Std Green', 'Std Blue',
                                'Entropy',
                                'Area', 'Perimeter', 'Circularity', 'Eccentricity'
                            ]
                            
                            if features_array.shape[1] == len(feature_names):
                                df_features = pd.DataFrame(features_array, columns=feature_names)
                                st.dataframe(df_features.style.highlight_max(axis=1))
                            
                                # Insight Logic (std_blue is now at index 5)
                                std_blue_val = features_array[0][5]
                                st.info(f"💡 **Insight:** `Std Blue` is **{std_blue_val:.4f}**. "
                                        f"We know from training that **{top_feat}** is the most critical feature.")
                            else:
                                st.write(features_array)

                        except Exception as e:
                            st.error(f"Error: {e}")
                        
                        if os.path.exists(temp_path):
                            os.remove(temp_path)

# --- MODE 2: FEATURE ANALYSIS ---
elif app_mode == "📊 Research: Feature Analysis":
    st.title("📊 Feature Analysis")
    st.markdown(f"Our research identified **{top_feat}** as the primary separator.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribution Analysis")
        # Generate distribution plot dynamically
        @st.cache_data
        def generate_distribution_plot():
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            try:
                df = pd.read_csv(os.path.join(project_root, 'data', 'galaxy_features.csv'))
            except FileNotFoundError:
                return None
            
            fig, ax = plt.subplots(figsize=(8, 5))
            
            # Plot distribution for each class
            spiral = df[df['label'] == 1]['std_blue']
            elliptical = df[df['label'] == 0]['std_blue']
            
            sns.histplot(spiral, kde=True, color='blue', label='Spiral', alpha=0.6, ax=ax)
            sns.histplot(elliptical, kde=True, color='red', label='Elliptical', alpha=0.6, ax=ax)
            
            ax.set_xlabel('std_blue')
            ax.set_ylabel('Count')
            ax.set_title(f'Distribution of std_blue by Galaxy Type')
            ax.legend()
            
            return fig
        
        fig = generate_distribution_plot()
        if fig:
            st.pyplot(fig)
        else:
            st.warning("Could not load data for distribution plot.")
            
    with col2:
        st.subheader("Feature Importance")
        show_plot_if_exists("feature_importance.png", "Top Features Selected by Trees")

# --- MODE 3: BOOSTING INTERNALS ---
elif app_mode == "🧠 Research: Boosting Internals":
    st.title("🧠 Boosting Internals")
    st.markdown("Visualizing the training process from `data/grid_search_detailed_logs.csv`.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Alpha Decay")
        show_plot_if_exists("alpha_decay.png", "Alpha (Weights) over iterations")
    with col2:
        st.subheader("Error Evolution")
        show_plot_if_exists("error_evolution.png", "Weighted Error over iterations")

# --- MODE 4: BIAS-VARIANCE ANALYSIS ---
elif app_mode == "⚖️ Research: Bias-Variance Analysis":
    st.title("⚖️ Bias-Variance Analysis")
    st.markdown("""
    This analysis compares **Boosting** vs **Bagging** to demonstrate why Boosting 
    is the right choice for weak learners (decision stumps).
    """)
    
    # Theory explanation
    st.markdown("### 📚 Key Concepts")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Bagging (Bootstrap Aggregating)**
        - Reduces **Variance** (overfitting)
        - Trains models on random subsets in parallel
        - Works best with **strong learners** (deep trees)
        - Averaging weak models = still weak
        """)
    
    with col2:
        st.info("""
        **Boosting (AdaBoost)**
        - Reduces **Bias** (underfitting)
        - Trains models sequentially, focusing on errors
        - Works best with **weak learners** (stumps)
        - Combining weak models = strong model
        """)
    
    st.markdown("---")
    st.markdown("### 🧪 Experiment Results")
    
    # Load pre-computed results from CSV (generated by 6_bias_variance_analysis.py)
    def load_bias_variance_results():
        csv_path = os.path.join(project_root, 'data', 'bias_variance_results.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            return dict(zip(df['Model'], df['Accuracy']))
        return None
    
    results = load_bias_variance_results()
    
    if results:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            acc = results.get('AdaBoost (Stumps)', 0)
            st.success("### 🚀 AdaBoost (Stumps)")
            st.markdown(f"## {acc*100:.2f}%")
            st.caption("Our approach - Boosting weak learners")
        
        with col2:
            acc = results.get('Bagging (Stumps)', 0)
            st.error("### ❌ Bagging (Stumps)")
            st.markdown(f"## {acc*100:.2f}%")
            st.caption("Bagging can't fix high bias")
        
        with col3:
            acc = results.get('Bagging (Deep Trees)', 0)
            st.info("### ✅ Bagging (Deep Trees)")
            st.markdown(f"## {acc*100:.2f}%")
            st.caption("Bagging works with strong learners")
        
        with col4:
            acc = results.get('Single Tree', 0)
            st.warning("### 🌳 Single Tree")
            st.markdown(f"## {acc*100:.2f}%")
            st.caption("Baseline comparison")
        
        st.markdown("---")
        st.markdown("### 💡 Key Insights")
        
        bagging_stumps = results.get('Bagging (Stumps)', 0)
        adaboost_stumps = results.get('AdaBoost (Stumps)', 0)
        bagging_deep = results.get('Bagging (Deep Trees)', 0)
        single_tree = results.get('Single Tree', 0)
        
        if bagging_stumps < adaboost_stumps:
            st.success(f"""
            ✅ **Boosting beats Bagging on Stumps** ({adaboost_stumps*100:.2f}% vs {bagging_stumps*100:.2f}%)
            
            Stumps have **high bias** (underfitting). Bagging reduces variance, not bias.
            Boosting sequentially corrects errors, effectively reducing bias.
            """)
        
        if bagging_deep > single_tree:
            st.info(f"""
            ✅ **Bagging improves Deep Trees** ({bagging_deep*100:.2f}% vs {single_tree*100:.2f}%)
            
            Deep trees have **high variance** (overfitting). Bagging averages out the variance,
            making the ensemble more stable than a single tree.
            """)
    else:
        st.error("Could not load data. Run `python src/6_bias_variance_analysis.py` first.")

# --- MODE 5: MODEL COMPARISON ---
elif app_mode == "🏆 Research: Model Comparison":
    st.title("🏆 Final Results")
    st.markdown("Comparison of our Manual AdaBoost against other algorithms.")
    
    show_plot_if_exists("advanced_comparison.png", "Accuracy Comparison")
    
    # Load pre-computed results from CSV (generated by 7_benchmark_comparison.py)
    def load_benchmark_results():
        csv_path = os.path.join(project_root, 'data', 'benchmark_comparison_results.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            return dict(zip(df['Model'], df['Accuracy']))
        return None
    
    comparison_results = load_benchmark_results()
    
    if comparison_results:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            acc = comparison_results.get("AdaBoost (Baseline)", 0)
            st.success("### AdaBoost (Baseline)")
            st.markdown(f"**{acc*100:.2f}%**")
        with col2:
            acc = comparison_results.get("Random Forest", 0)
            st.info("### Random Forest")
            st.markdown(f"**{acc*100:.2f}%**")
        with col3:
            acc = comparison_results.get("Gradient Boosting", 0)
            st.warning("### Gradient Boosting")
            st.markdown(f"**{acc*100:.2f}%**")
        with col4:
            acc = comparison_results.get("SVM (RBF Kernel)", 0)
            st.error("### SVM (RBF Kernel)")
            st.markdown(f"**{acc*100:.2f}%**")
    else:
        st.error("Could not load data. Run `python src/7_benchmark_comparison.py` first.")

# Footer
st.markdown("---")
st.caption("Built with Python & Custom AdaBoost Implementation")