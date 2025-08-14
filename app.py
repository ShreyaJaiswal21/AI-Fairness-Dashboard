import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from fairlearn.reductions import GridSearch, DemographicParity, EqualizedOdds

# --- Page Configuration ---
st.set_page_config(
    page_title="Algorithmic Fairness Dashboard",
    page_icon="⚖️",
    layout="wide"
)

# --- Data Loading and Preprocessing ---
@st.cache_data
def load_and_prep_data():
    """Loads, splits, and preprocesses the Adult dataset."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
    df = pd.read_csv(url, names=names, na_values='?', skipinitialspace=True).dropna()
    
    X = df.drop(columns='income')
    y = LabelEncoder().fit_transform(df['income'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=np.number).columns
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop'
    )
    
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    return X_train, X_test, y_train, y_test, X_train_transformed, X_test_transformed, preprocessor

# --- Model Training Functions (Cached) ---
@st.cache_resource
def train_models(X_train_transformed, y_train, _X_train): # _X_train is used for sensitive features
    """Trains and returns the baseline and mitigated models."""
    # Train baseline model
    baseline_model = LogisticRegression(solver='liblinear', random_state=42).fit(X_train_transformed, y_train)
    
    # Train Demographic Parity mitigator
    dp_mitigator = GridSearch(
        LogisticRegression(solver='liblinear', random_state=42),
        constraints=DemographicParity(),
        grid_size=40
    )
    dp_mitigator.fit(X_train_transformed, y_train, sensitive_features=_X_train['sex'])
    
    # Train Equalized Odds mitigator
    eo_mitigator = GridSearch(
        LogisticRegression(solver='liblinear', random_state=42),
        constraints=EqualizedOdds(),
        grid_size=40
    )
    eo_mitigator.fit(X_train_transformed, y_train, sensitive_features=_X_train['sex'])
    
    return baseline_model, dp_mitigator, eo_mitigator

# --- Main App ---
st.title("⚖️ Interactive Algorithmic Fairness Dashboard")
st.write("""
This dashboard demonstrates how to audit a machine learning model for bias and apply mitigation techniques using the Fairlearn library.
We are analyzing the 'Adult' income dataset to predict whether an individual earns more than $50K/year, with 'sex' as the sensitive feature.
""")

# Load data and get sensitive features from the original (untransformed) test set
X_train, X_test, y_train, y_test, X_train_transformed, X_test_transformed, preprocessor = load_and_prep_data()
sensitive_features_test = X_test['sex']

# --- Sidebar for controls ---
st.sidebar.header("Analysis Controls")
run_analysis = st.sidebar.button("Show Fairness Analysis")

# --- Main content area ---
if run_analysis:
    # Train models (will be cached after first run)
    with st.spinner("Training models... This only happens once."):
        baseline_model, dp_mitigator, eo_mitigator = train_models(X_train_transformed, y_train, X_train)

    # --- Baseline Model ---
    st.header("1. Baseline Model Performance")
    with st.spinner("Analyzing baseline model..."):
        y_pred_baseline = baseline_model.predict(X_test_transformed)
        baseline_accuracy = accuracy_score(y_test, y_pred_baseline)
        baseline_dp_diff = demographic_parity_difference(y_true=y_test, y_pred=y_pred_baseline, sensitive_features=sensitive_features_test)
        baseline_eo_diff = equalized_odds_difference(y_true=y_test, y_pred=y_pred_baseline, sensitive_features=sensitive_features_test)

    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Accuracy", f"{baseline_accuracy:.4f}")
    col2.metric("Demographic Parity Difference", f"{baseline_dp_diff:.4f}", help="Difference in selection rate between groups. Closer to 0 is fairer.")
    col3.metric("Equalized Odds Difference", f"{baseline_eo_diff:.4f}", help="Difference in error rates (true/false positives) between groups. Closer to 0 is fairer.")
    
    # --- Visualization ---
    st.header("2. Accuracy-Fairness Trade-off Visualization")
    st.info("This plot shows the performance of all models evaluated by Fairlearn's GridSearch. You can hover over any point to see its specific metrics.")

    def get_model_metrics(mitigator, X_test_data, y_test_data, sensitive_test_data):
        """Helper to calculate metrics for all models in a GridSearch result."""
        accuracies, dp_diffs, eo_diffs = [], [], []
        for model in mitigator.predictors_:
            y_pred = model.predict(X_test_data)
            accuracies.append(accuracy_score(y_test_data, y_pred))
            dp_diffs.append(demographic_parity_difference(y_true=y_test_data, y_pred=y_pred, sensitive_features=sensitive_test_data))
            eo_diffs.append(equalized_odds_difference(y_true=y_test_data, y_pred=y_pred, sensitive_features=sensitive_test_data))
        return accuracies, dp_diffs, eo_diffs

    with st.spinner("Generating visualization..."):
        dp_accuracies, dp_dp_diffs, dp_eo_diffs = get_model_metrics(dp_mitigator, X_test_transformed, y_test, sensitive_features_test)
        eo_accuracies, eo_dp_diffs, eo_eo_diffs = get_model_metrics(eo_mitigator, X_test_transformed, y_test, sensitive_features_test)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dp_dp_diffs, y=dp_accuracies, mode='markers', name='Demographic Parity Models', marker=dict(color='blue', opacity=0.7), hovertemplate='<b>DP Model</b><br>Accuracy: %{y:.4f}<br>DP Difference: %{x:.4f}<br>EO Difference: %{customdata:.4f}<extra></extra>', customdata=np.array(dp_eo_diffs)))
        fig.add_trace(go.Scatter(x=eo_dp_diffs, y=eo_accuracies, mode='markers', name='Equalized Odds Models', marker=dict(color='orange', opacity=0.7, symbol='diamond'), hovertemplate='<b>EO Model</b><br>Accuracy: %{y:.4f}<br>DP Difference: %{x:.4f}<br>EO Difference: %{customdata:.4f}<extra></extra>', customdata=np.array(eo_eo_diffs)))
        fig.add_trace(go.Scatter(x=[baseline_dp_diff], y=[baseline_accuracy], mode='markers', name='Baseline Model', marker=dict(color='red', size=15, symbol='star'), hovertemplate='<b>Baseline Model</b><br>Accuracy: %{y:.4f}<br>DP Difference: %{x:.4f}<br>EO Difference: %{customdata:.4f}<extra></extra>', customdata=np.array([baseline_eo_diff])))
        fig.update_layout(title='Interactive Accuracy vs. Fairness Trade-off Analysis', xaxis_title='Demographic Parity Difference (Lower is Fairer)', yaxis_title='Overall Accuracy', legend_title='Model Type', template='plotly_white', height=600)
        
        st.plotly_chart(fig, use_container_width=True)
        
else:
    st.info("Click the 'Show Fairness Analysis' button in the sidebar to begin.")
