import streamlit as st
import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import json
import pickle
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Explainability AI Tool (Noor)",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'datasets' not in st.session_state:
    st.session_state.datasets = {}
if 'comparisons' not in st.session_state:
    st.session_state.comparisons = {}

def main():
    st.title("🔍 Explainability AI Tool (Project Noor)")
    st.markdown("""
    A comprehensive system for explaining and comparing Machine Learning/AI models used by banks.
    This tool enables central banks to understand, compare, and supervise the AI models used by different financial institutions.
    """)
    
    # Create tabs for Bank and Supervisor views
    view_mode = st.radio("Select View:", ["Bank View", "Supervisor View"], horizontal=True)
    
    if view_mode == "Bank View":
        show_bank_view()
    else:
        show_supervisor_view()

def show_home():
    st.header("Welcome to the Explainability AI Tool")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Purpose")
        st.write("""
        - Provide explainability for ML/AI models used by banks
        - Enable central banks to compare models across different banks
        - Ensure transparency in AI-driven financial decisions
        - Support regulatory oversight of AI applications in banking
        """)
    
    with col2:
        st.subheader("Key Features")
        st.write("""
        - Model explainability engine with SHAP and LIME
        - Cross-bank comparison framework
        - Regulatory compliance module
        - Privacy-preserving analysis
        - Interactive visualization dashboard
        """)
    
    st.subheader("How to Use")
    st.write("""
    1. **Upload Data**: Upload your dataset to analyze
    2. **Train Model**: Train an ML model on your data
    3. **Model Explainability**: Understand how your model makes decisions
    4. **Cross-Bank Comparison**: Compare models across different institutions
    5. **Regulatory Compliance**: Check compliance with regulations
    6. **Privacy Analysis**: Analyze privacy-preserving capabilities
    """)

def show_upload_data():
    st.header("Upload Dataset")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload a dataset to analyze with the explainability tools"
    )
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.datasets[uploaded_file.name] = df
        
        st.success(f"Dataset '{uploaded_file.name}' uploaded successfully!")
        st.write(f"Shape: {df.shape}")
        
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        st.subheader("Data Summary")
        st.write(df.describe())

def show_train_model():
    st.header("Train Model")
    
    if not st.session_state.datasets:
        st.warning("Please upload a dataset first in the 'Upload Data' section.")
        return
    
    dataset_name = st.selectbox("Select dataset:", list(st.session_state.datasets.keys()))
    df = st.session_state.datasets[dataset_name]
    
    st.subheader("Select Target Variable")
    target_col = st.selectbox("Choose target column:", df.columns.tolist())
    
    st.subheader("Select Features")
    feature_cols = st.multiselect(
        "Choose feature columns:",
        [col for col in df.columns if col != target_col],
        default=[col for col in df.columns if col != target_col][:5]  # Default to first 5 features
    )
    
    if not feature_cols:
        st.warning("Please select at least one feature column.")
        return
    
    # Prepare data
    X = df[feature_cols]
    y = df[target_col]
    
    # Handle categorical variables
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    
    st.subheader("Model Selection")
    model_type = st.selectbox("Choose model type:", ["Random Forest", "Logistic Regression"])
    
    if st.button("Train Model"):
        if model_type == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = LogisticRegression(random_state=42)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        st.success(f"Model trained successfully! Accuracy: {accuracy:.2f}")
        
        # Save model
        model_name = f"{model_type}_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        st.session_state.models[model_name] = {
            'model': model,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': X_encoded.columns.tolist(),
            'target_name': target_col
        }
        
        st.write(f"Model saved as: {model_name}")

def show_model_explainability():
    st.header("Model Explainability")
    
    if not st.session_state.models:
        st.warning("Please train a model first in the 'Train Model' section.")
        return
    
    model_name = st.selectbox("Select model:", list(st.session_state.models.keys()))
    model_info = st.session_state.models[model_name]
    model = model_info['model']
    X_test = model_info['X_test']
    feature_names = model_info['feature_names']
    
    st.subheader(f"Model: {model_name}")
    
    # Project Veritas-inspired validation framework
    st.info("🔬 Based on Project Veritas: Explainable AI Validation Framework by MAS")
    
    explanation_method = st.radio(
        "Choose explanation method:",
        ["SHAP", "LIME", "Feature Importance", "Project Veritas Validation"]
    )
    
    if explanation_method == "SHAP":
        show_shap_explanation(model, X_test, feature_names)
    elif explanation_method == "LIME":
        show_lime_explanation(model, X_test, feature_names)
    elif explanation_method == "Feature Importance":
        show_feature_importance(model, feature_names)
    elif explanation_method == "Project Veritas Validation":
        show_project_veritas_validation(model, X_test, feature_names, model_info)

def show_project_veritas_validation(model, X_test, feature_names, model_info):
    st.subheader("Project Veritas Validation Framework")
    
    st.write("""
    Project Veritas by MAS provides a validation framework for Explainable AI in financial services.
    This includes:
    - **Explainability**: Understanding how models make decisions
    - **Fairness**: Ensuring models don't discriminate unfairly
    - **Performance**: Validating model effectiveness
    - **Robustness**: Ensuring model reliability under various conditions
    """)
    
    # Create tabs for different validation aspects
    tab1, tab2, tab3, tab4 = st.tabs(["Explainability", "Fairness", "Performance", "Robustness"])
    
    with tab1:
        st.header("Explainability Assessment")
        st.write("Analyzing how the model arrives at its decisions using multiple explanation methods:")
        
        # SHAP explanation
        st.subheader("SHAP Analysis")
        try:
            explainer = shap.TreeExplainer(model) if hasattr(model, 'tree') else shap.LinearExplainer(model, X_test)
            shap_values = explainer.shap_values(X_test.iloc[:50])  # Use subset for performance
            
            # Handle binary classification
            if len(shap_values.shape) == 2 and shap_values.shape[1] == len(feature_names):
                shap_vals = shap_values
            else:
                shap_vals = shap_values[1] if isinstance(shap_values, list) else shap_values
            
            # Summary plot
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_vals, X_test.iloc[:50], feature_names=feature_names, show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.clf()
        except:
            st.warning("SHAP analysis not available for this model type")
        
        # Feature importance
        st.subheader("Feature Importance")
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig = px.bar(importance_df, x='importance', y='feature', orientation='h',
                         title="Feature Importance", labels={'importance': 'Importance', 'feature': 'Feature'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance not available for this model type")
    
    with tab2:
        st.header("Fairness Assessment")
        st.write("Evaluating model fairness across different groups (simulated analysis):")
        
        # Simulate fairness analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Demographic Parity Check")
            st.write("Ensuring equal positive prediction rates across groups")
            
            # Simulate group analysis
            groups = ['Group A', 'Group B', 'Group C']
            pos_rates = [0.62, 0.58, 0.65]  # Simulated positive prediction rates
            
            fairness_df = pd.DataFrame({
                'Group': groups,
                'Positive Rate': pos_rates
            })
            
            fig = px.bar(fairness_df, x='Group', y='Positive Rate', 
                         title="Positive Prediction Rate by Group",
                         range_y=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Equal Opportunity Check")
            st.write("Ensuring equal true positive rates across groups")
            
            tpr_rates = [0.71, 0.68, 0.73]  # Simulated true positive rates
            opp_df = pd.DataFrame({
                'Group': groups,
                'True Positive Rate': tpr_rates
            })
            
            fig = px.bar(opp_df, x='Group', y='True Positive Rate',
                         title="True Positive Rate by Group",
                         range_y=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Performance Assessment")
        st.write("Evaluating model performance metrics:")
        
        # Calculate performance metrics
        y_pred = model.predict(X_test)
        y_true = model_info['y_test'][:len(y_pred)]  # Align with predictions
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            accuracy = accuracy_score(y_true, y_pred)
            st.metric("Accuracy", f"{accuracy:.3f}")
        
        with col2:
            from sklearn.metrics import precision_score, recall_score
            try:
                precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                st.metric("Precision", f"{precision:.3f}")
            except:
                st.metric("Precision", "N/A")
        
        with col3:
            try:
                recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                st.metric("Recall", f"{recall:.3f}")
            except:
                st.metric("Recall", "N/A")
        
        # ROC curve for binary classification
        if len(np.unique(y_true)) == 2:
            from sklearn.metrics import roc_curve, auc
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            fig = px.line(x=fpr, y=tpr, title=f"ROC Curve (AUC = {roc_auc:.3f})")
            fig.add_shape(type='line', x0=0, x1=1, y0=0, y1=1, line=dict(dash='dash'))
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("Robustness Assessment")
        st.write("Testing model reliability under various conditions:")
        
        # Perturbation analysis
        st.subheader("Input Perturbation Test")
        st.write("How model predictions change with small input variations")
        
        if len(X_test) > 0:
            sample_idx = st.slider("Select sample to perturb:", 0, min(len(X_test)-1, 10), 0)
            original_sample = X_test.iloc[sample_idx].copy()
            original_pred = model.predict([original_sample])[0]
            
            st.write(f"Original prediction: {original_pred}")
            
            # Add small perturbations
            perturbed_samples = []
            predictions = []
            
            for i in range(10):
                perturbation = np.random.normal(0, 0.01, size=original_sample.shape)  # 1% std deviation
                perturbed_sample = original_sample + perturbation
                perturbed_sample = pd.DataFrame([perturbed_sample], columns=X_test.columns)
                
                pred = model.predict(perturbed_sample)[0]
                perturbed_samples.append(i)
                predictions.append(pred)
            
            robustness_df = pd.DataFrame({
                'Perturbation': perturbed_samples,
                'Prediction': predictions
            })
            
            fig = px.line(robustness_df, x='Perturbation', y='Prediction',
                          title="Prediction Stability Under Input Perturbation")
            st.plotly_chart(fig, use_container_width=True)
            
            stability = 1 - (len(set(predictions)) / len(predictions))
            st.metric("Prediction Stability", f"{stability:.2%}")

def show_shap_explanation(model, X_test, feature_names):
    st.subheader("SHAP Explanation")
    
    # Calculate SHAP values
    explainer = shap.TreeExplainer(model) if hasattr(model, 'tree') else shap.LinearExplainer(model, X_test)
    shap_values = explainer.shap_values(X_test.iloc[:100])  # Use subset for performance
    
    # Handle binary classification
    if len(shap_values.shape) == 2 and shap_values.shape[1] == len(feature_names):
        # Binary classification - use values for positive class
        shap_vals = shap_values
    else:
        # Multi-class or other format - handle appropriately
        shap_vals = shap_values[1] if isinstance(shap_values, list) else shap_values
    
    # Summary plot
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_vals, X_test.iloc[:100], feature_names=feature_names, show=False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.clf()
    
    # Individual prediction explanation
    st.subheader("Individual Prediction Explanation")
    sample_idx = st.slider("Select sample to explain:", 0, min(len(X_test)-1, 99), 0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(explainer.expected_value, shap_vals[sample_idx], X_test.iloc[sample_idx], show=False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.clf()

def show_lime_explanation(model, X_test, feature_names):
    st.subheader("LIME Explanation")
    
    # Create LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_test.values,
        feature_names=feature_names,
        class_names=['Negative', 'Positive'],  # Adjust as needed
        verbose=True,
        mode='classification'
    )
    
    sample_idx = st.slider("Select sample to explain:", 0, len(X_test)-1, 0)
    sample = X_test.iloc[sample_idx].values
    
    # Explain instance
    exp = explainer.explain_instance(sample, model.predict_proba, num_features=len(feature_names))
    
    # Display explanation
    exp.show_in_notebook(show_table=True, show_all=False)
    
    # Alternative: Show as dataframe
    lime_exp = exp.as_list()
    lime_df = pd.DataFrame(lime_exp, columns=['Feature', 'Contribution'])
    st.dataframe(lime_df)

def show_feature_importance(model, feature_names):
    st.subheader("Feature Importance")
    
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig = px.bar(importance_df, x='importance', y='feature', orientation='h',
                     title="Feature Importance", labels={'importance': 'Importance', 'feature': 'Feature'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Selected model does not support feature importance.")

def show_cross_bank_comparison():
    st.header("Cross-Bank Model Comparison")
    
    if len(st.session_state.models) < 2:
        st.warning("Please train at least 2 models to enable comparison.")
        return
    
    model_names = list(st.session_state.models.keys())
    selected_models = st.multiselect("Select models to compare:", model_names, default=model_names[:2])
    
    if len(selected_models) < 2:
        st.warning("Please select at least 2 models for comparison.")
        return
    
    st.subheader("Model Performance Comparison")
    
    comparison_data = []
    for name in selected_models:
        model_info = st.session_state.models[name]
        model = model_info['model']
        X_test = model_info['X_test']
        y_test = model_info['y_test']
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        comparison_data.append({
            'Model': name,
            'Accuracy': accuracy,
            'Features': len(model_info['feature_names']),
            'Target': model_info['target_name']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df)
    
    # Visualize comparison
    fig = px.bar(comparison_df, x='Model', y='Accuracy', 
                 title="Model Accuracy Comparison", 
                 labels={'Accuracy': 'Accuracy Score', 'Model': 'Model Name'})
    st.plotly_chart(fig, use_container_width=True)

def show_regulatory_compliance():
    st.header("Regulatory Compliance Check")
    
    if not st.session_state.models:
        st.warning("Please train a model first.")
        return
    
    model_name = st.selectbox("Select model for compliance check:", list(st.session_state.models.keys()))
    model_info = st.session_state.models[model_name]
    
    st.subheader(f"Compliance Analysis for: {model_name}")
    
    # Fair lending analysis
    st.subheader("Fair Lending Analysis")
    st.write("This section would analyze the model for potential bias based on protected attributes.")
    
    # Model documentation
    st.subheader("Model Documentation")
    st.write("**Model Type:**", type(model_info['model']).__name__)
    st.write("**Features Used:**", len(model_info['feature_names']))
    st.write("**Target Variable:**", model_info['target_name'])
    st.write("**Training Date:**", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Download documentation
    doc_data = {
        "model_name": model_name,
        "model_type": type(model_info['model']).__name__,
        "features": model_info['feature_names'],
        "target_variable": model_info['target_name'],
        "analysis_date": datetime.now().isoformat(),
        "compliance_status": "PENDING"  # Placeholder
    }
    
    st.download_button(
        label="Download Compliance Report",
        data=json.dumps(doc_data, indent=2),
        file_name=f"compliance_report_{model_name}.json",
        mime="application/json"
    )

def show_privacy_analysis():
    st.header("Privacy Analysis")
    
    st.subheader("Privacy-Preserving Techniques")
    st.write("""
    This section demonstrates privacy-preserving analysis techniques:
    
    - **Differential Privacy**: Adding noise to data to preserve privacy
    - **Homomorphic Encryption**: Performing computations on encrypted data
    - **Secure Multi-Party Computation**: Computing joint functions without revealing inputs
    """)
    
    if st.session_state.models:
        model_name = st.selectbox("Select model for privacy analysis:", list(st.session_state.models.keys()))
        st.write(f"Analyzing privacy features for: {model_name}")
        
        # Placeholder for privacy analysis
        st.info("Privacy analysis features would be implemented here to demonstrate how models can be analyzed without exposing sensitive data.")

def show_bank_view():
    st.header("🏦 Bank View - Model Development & Management")
    
    # Bank-specific sidebar navigation
    st.sidebar.header("Bank Navigation")
    page = st.sidebar.selectbox(
        "Choose a function:",
        ["Dashboard", "Upload Data", "Train Credit Risk Model", "Model Explainability", "Internal Compliance"]
    )
    
    if page == "Dashboard":
        show_bank_dashboard()
    elif page == "Upload Data":
        show_upload_data()
    elif page == "Train Credit Risk Model":
        show_train_credit_risk_model()
    elif page == "Model Explainability":
        show_model_explainability()
    elif page == "Internal Compliance":
        show_internal_compliance()

def show_supervisor_view():
    st.header("🏛️ Supervisor View - Model Oversight & Comparison")
    
    # Supervisor-specific sidebar navigation
    st.sidebar.header("Supervisor Navigation")
    page = st.sidebar.selectbox(
        "Choose a function:",
        ["Supervisor Dashboard", "Model Comparison", "Cross-Bank Analysis", "Regulatory Compliance", "Risk Assessment"]
    )
    
    if page == "Supervisor Dashboard":
        show_supervisor_dashboard()
    elif page == "Model Comparison":
        show_cross_bank_comparison()
    elif page == "Cross-Bank Analysis":
        show_cross_bank_analysis()
    elif page == "Regulatory Compliance":
        show_regulatory_compliance()
    elif page == "Risk Assessment":
        show_risk_assessment()

def show_bank_dashboard():
    st.subheader("Bank Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Active Models", value="3", delta="+1")
    with col2:
        st.metric(label="Models in Review", value="1", delta="-1")
    with col3:
        st.metric(label="Compliance Score", value="87%", delta="+5%")
    
    st.subheader("Recent Activity")
    st.write("- Credit risk model v2.1 submitted for review")
    st.write("- New loan application data ingested (50K records)")
    st.write("- Feature importance report generated")
    
    st.subheader("Quick Actions")
    if st.button("Train New Credit Risk Model"):
        st.session_state.active_tab = "Train Credit Risk Model"
    if st.button("Generate Explainability Report"):
        st.session_state.active_tab = "Model Explainability"

def show_supervisor_dashboard():
    st.subheader("Supervisor Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Banks Monitored", value="12", delta="+2")
    with col2:
        st.metric(label="Models Under Review", value="18", delta="+3")
    with col3:
        st.metric(label="Compliance Rate", value="78%", delta="-2%")
    with col4:
        st.metric(label="Risk Alerts", value="5", delta="+1")
    
    st.subheader("Systemic Risk Indicators")
    # Create a simple risk indicator visualization
    risk_data = pd.DataFrame({
        'Indicator': ['Credit Growth', 'Asset Quality', 'Capital Adequacy', 'Market Risk'],
        'Score': [75, 82, 90, 68]
    })
    fig = px.bar(risk_data, x='Indicator', y='Score', 
                 title="Risk Indicators Across Banks",
                 range_y=[0, 100],
                 color='Score',
                 color_continuous_scale=['green', 'yellow', 'red'])
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Recent Submissions")
    st.write("- Bank A: Credit risk model update")
    st.write("- Bank B: New fraud detection model")
    st.write("- Bank C: Model explainability report")

def show_train_credit_risk_model():
    st.header("Train Credit Risk Model")
    
    st.info("This demo shows how a bank might train a credit risk model using customer data.")
    
    # Generate synthetic credit risk data
    if st.button("Generate Demo Credit Risk Data"):
        np.random.seed(42)
        n_samples = 1000
        
        # Create synthetic features
        age = np.random.normal(40, 15, n_samples)
        income = np.random.lognormal(10, 0.5, n_samples)
        loan_amount = np.random.lognormal(9, 0.8, n_samples)
        credit_score = np.random.normal(650, 100, n_samples)
        employment_years = np.random.gamma(2, 2, n_samples)
        
        # Create synthetic target (loan default: 1 = default, 0 = no default)
        default_prob = (
            0.1 +
            0.3 * (credit_score < 600) +
            0.2 * (income < 30000) +
            0.15 * (loan_amount / income > 0.3) +
            0.1 * (employment_years < 2) +
            np.random.normal(0, 0.1, n_samples)
        )
        default = (np.random.random(n_samples) < default_prob).astype(int)
        
        # Create DataFrame
        df = pd.DataFrame({
            'age': age,
            'income': income,
            'loan_amount': loan_amount,
            'credit_score': credit_score,
            'employment_years': employment_years,
            'default': default
        })
        
        # Round values
        df['age'] = df['age'].round().astype(int)
        df['income'] = df['income'].round(-2)  # Round to hundreds
        df['loan_amount'] = df['loan_amount'].round(-2)
        df['credit_score'] = df['credit_score'].round().astype(int)
        df['employment_years'] = df['employment_years'].round(1)
        
        st.session_state.datasets['demo_credit_risk.csv'] = df
        st.success("Demo credit risk data generated and loaded!")
        
        st.subheader("Sample of Generated Data")
        st.dataframe(df.head())
        
        # Show data statistics
        st.subheader("Data Statistics")
        st.write(df.describe())
        
        # Show target distribution
        st.subheader("Default Distribution")
        default_counts = df['default'].value_counts()
        st.write(f"Non-default: {default_counts[0]} ({default_counts[0]/len(df)*100:.1f}%)")
        st.write(f"Default: {default_counts[1]} ({default_counts[1]/len(df)*100:.1f}%)")

def show_cross_bank_analysis():
    st.header("Cross-Bank Model Analysis")
    
    st.info("This view allows supervisors to compare models across different banks.")
    
    # Simulate having models from different banks
    banks = ["Bank A", "Bank B", "Bank C", "Bank D"]
    
    if st.button("Generate Demo Bank Models"):
        # Create demo models for different banks
        for i, bank in enumerate(banks):
            # Generate synthetic model data
            np.random.seed(42 + i)
            n_samples = 500
            features = [f"feature_{j}" for j in range(5)]
            
            # Create dummy data
            X = pd.DataFrame({f: np.random.randn(n_samples) for f in features})
            y = (0.3*X.iloc[:, 0] + 0.2*X.iloc[:, 1] + 0.1*np.random.randn(n_samples) > 0).astype(int)
            
            # Train a simple model
            model = LogisticRegression()
            model.fit(X, y)
            
            # Save the model
            model_name = f"{bank}_credit_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            st.session_state.models[model_name] = {
                'model': model,
                'X_train': X,
                'X_test': X,
                'y_train': y,
                'y_test': y,
                'feature_names': features,
                'target_name': 'default_risk',
                'bank_name': bank
            }
        
        st.success(f"Demo models created for {len(banks)} banks!")
    
    if st.session_state.models:
        # Group models by bank
        bank_models = {}
        for model_name, model_info in st.session_state.models.items():
            bank_name = model_info.get('bank_name', 'Unknown')
            if bank_name not in bank_models:
                bank_models[bank_name] = []
            bank_models[bank_name].append((model_name, model_info))
        
        st.subheader("Models by Bank")
        for bank_name, models in bank_models.items():
            with st.expander(f"📁 {bank_name} ({len(models)} models)"):
                for model_name, model_info in models:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**Model:** {model_name}")
                    with col2:
                        st.write(f"**Features:** {len(model_info['feature_names'])}")
                    with col3:
                        st.write(f"**Target:** {model_info['target_name']}")

def show_risk_assessment():
    st.header("Risk Assessment Dashboard")
    
    st.info("Centralized view for assessing risks across all monitored banks.")
    
    # Create a mock risk assessment dashboard
    banks = ["Bank A", "Bank B", "Bank C", "Bank D"]
    risk_categories = ["Credit Risk", "Market Risk", "Operational Risk", "Liquidity Risk"]
    
    # Generate random risk scores
    risk_data = []
    for bank in banks:
        for category in risk_categories:
            risk_data.append({
                'Bank': bank,
                'Risk Category': category,
                'Risk Score': np.random.randint(1, 100)
            })
    
    risk_df = pd.DataFrame(risk_data)
    
    # Create heatmap
    pivot_risk = risk_df.pivot(index='Bank', columns='Risk Category', values='Risk Score')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot_risk, annot=True, fmt='.0f', cmap='RdYlGn_r', center=50, ax=ax)
    ax.set_title("Risk Assessment Heatmap (Red=High Risk, Green=Low Risk)")
    st.pyplot(fig)
    plt.clf()
    
    # Show high-risk items
    high_risk = risk_df[risk_df['Risk Score'] > 70]
    if not high_risk.empty:
        st.subheader("⚠️ High-Risk Items (>70)")
        st.dataframe(high_risk.sort_values('Risk Score', ascending=False))

def show_internal_compliance():
    st.header("Internal Compliance Check")
    
    st.info("Bank's internal compliance review before submitting to supervisors.")
    
    if st.session_state.models:
        model_name = st.selectbox("Select model for internal compliance:", list(st.session_state.models.keys()))
        
        if st.button("Run Internal Compliance Check"):
            st.subheader(f"Compliance Report for: {model_name}")
            
            # Generate mock compliance report
            st.write("### Model Documentation")
            st.write("- ✓ Model purpose clearly defined")
            st.write("- ✓ Data lineage documented")
            st.write("- ✓ Feature engineering explained")
            st.write("- ✓ Model architecture documented")
            
            st.write("### Bias Assessment")
            st.write("- ⚠ Need to assess fairness across demographic groups")
            st.write("- ⚠ Need to validate against protected attributes")
            
            st.write("### Performance Validation")
            st.write("- ✓ Train/validation/test split properly implemented")
            st.write("- ✓ Performance metrics calculated")
            st.write("- ⚠ Need out-of-time validation")
            
            st.write("### Risk Assessment")
            st.write("- ✓ Model risk tier assigned")
            st.write("- ⚠ Need stress testing scenarios")
            
            st.success("Internal compliance check completed. Ready for supervisor submission.")

if __name__ == "__main__":
    main()