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
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix
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
if 'test_results' not in st.session_state:
    st.session_state.test_results = {}
if 'use_cases' not in st.session_state:
    st.session_state.use_cases = [
        "Credit Risk", 
        "Fraud Detection", 
        "Anti-Money Laundering", 
        "Market Risk", 
        "Operational Risk",
        "Customer Churn Prediction",
        "Loan Default Prediction",
        "Insurance Claim Prediction"
    ]

def main():
    st.title("🔍 Explainability AI Tool")
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
    
    st.info("Compare models across different banks and use cases.")
    
    # Define use cases
    use_cases = ["Credit Risk", "Fraud Detection", "Anti-Money Laundering", "Market Risk", "Operational Risk"]
    selected_use_case = st.selectbox("Select Use Case:", use_cases)
    
    if st.button(f"Generate Demo {selected_use_case} Models"):
        # Simulate generating models for different banks for the selected use case
        banks = ["Bank A", "Bank B", "Bank C", "Bank D"]
        
        for i, bank in enumerate(banks):
            # Generate synthetic model data based on use case
            np.random.seed(42 + i)
            n_samples = 500
            
            if selected_use_case == "Credit Risk":
                # Credit risk features
                features = ['age', 'income', 'loan_amount', 'credit_score', 'employment_years']
                X = pd.DataFrame({
                    'age': np.random.normal(40, 15, n_samples),
                    'income': np.random.lognormal(10, 0.5, n_samples),
                    'loan_amount': np.random.lognormal(9, 0.8, n_samples),
                    'credit_score': np.random.normal(650, 100, n_samples),
                    'employment_years': np.random.gamma(2, 2, n_samples)
                })
                # Create target based on risk factors
                default_prob = (
                    0.1 +
                    0.3 * (X['credit_score'] < 600) +
                    0.2 * (X['income'] < 30000) +
                    0.15 * (X['loan_amount'] / X['income'] > 0.3) +
                    0.1 * (X['employment_years'] < 2) +
                    np.random.normal(0, 0.1, n_samples)
                )
                y = (np.random.random(n_samples) < default_prob).astype(int)
                
            elif selected_use_case == "Fraud Detection":
                # Fraud detection features
                features = ['transaction_amount', 'account_age_days', 'num_transactions_day', 'merchant_risk_score', 'time_since_last_transaction']
                X = pd.DataFrame({
                    'transaction_amount': np.random.lognormal(8, 1.5, n_samples),
                    'account_age_days': np.random.exponential(365, n_samples),
                    'num_transactions_day': np.random.poisson(5, n_samples),
                    'merchant_risk_score': np.random.uniform(0, 1, n_samples),
                    'time_since_last_transaction': np.random.exponential(2, n_samples)
                })
                # Create target based on fraud indicators
                fraud_prob = (
                    0.05 +
                    0.2 * (X['transaction_amount'] > 5000) +
                    0.1 * (X['merchant_risk_score'] > 0.8) +
                    0.15 * (X['num_transactions_day'] > 10) +
                    np.random.normal(0, 0.05, n_samples)
                )
                y = (np.random.random(n_samples) < fraud_prob).astype(int)
                
            else:
                # Generic features for other use cases
                features = [f"feature_{j}" for j in range(5)]
                X = pd.DataFrame({f: np.random.randn(n_samples) for f in features})
                y = (0.3*X.iloc[:, 0] + 0.2*X.iloc[:, 1] + 0.1*np.random.randn(n_samples) > 0).astype(int)
            
            # Train a simple model
            model = LogisticRegression(max_iter=1000)
            model.fit(X, y)
            
            # Save the model
            model_name = f"{bank}_{selected_use_case.lower().replace(' ', '_')}_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            st.session_state.models[model_name] = {
                'model': model,
                'X_train': X,
                'X_test': X,
                'y_train': y,
                'y_test': y,
                'feature_names': features,
                'target_name': 'risk_outcome',
                'bank_name': bank,
                'use_case': selected_use_case
            }
        
        st.success(f"Demo {selected_use_case} models created for {len(banks)} banks!")
    
    if st.session_state.models:
        # Filter models by selected use case
        filtered_models = {k: v for k, v in st.session_state.models.items() 
                          if v.get('use_case', '') == selected_use_case}
        
        if filtered_models:
            # Group models by bank
            bank_models = {}
            for model_name, model_info in filtered_models.items():
                bank_name = model_info.get('bank_name', 'Unknown')
                if bank_name not in bank_models:
                    bank_models[bank_name] = []
                bank_models[bank_name].append((model_name, model_info))
            
            st.subheader(f"Models by Bank - {selected_use_case}")
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
            
            # Performance comparison if we have test data
            if st.button("Compare Model Performance"):
                comparison_data = []
                for model_name, model_info in filtered_models.items():
                    model = model_info['model']
                    X_test = model_info['X_test']
                    y_test = model_info['y_test']
                    
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    comparison_data.append({
                        'Model': model_name,
                        'Bank': model_info['bank_name'],
                        'Accuracy': accuracy,
                        'Features': len(model_info['feature_names']),
                        'Use Case': model_info['use_case']
                    })
                
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    st.subheader(f"Performance Comparison - {selected_use_case}")
                    st.dataframe(comparison_df)
                    
                    # Visualize comparison
                    fig = px.bar(comparison_df, x='Model', y='Accuracy', 
                                color='Bank',
                                title=f"Model Accuracy Comparison - {selected_use_case}",
                                labels={'Accuracy': 'Accuracy Score', 'Model': 'Model'})
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No {selected_use_case} models available. Generate demo models to see comparison.")
    else:
        st.info("No models available. Generate demo models to see comparison.")

def show_regulatory_compliance():
    st.header("Regulatory Compliance Check")
    
    if not st.session_state.models:
        st.warning("Please train a model first.")
        return
    
    model_name = st.selectbox("Select model for compliance check:", list(st.session_state.models.keys()))
    model_info = st.session_state.models[model_name]
    
    st.subheader(f"Compliance Analysis for: {model_name}")
    
    # Model documentation
    st.subheader("Model Documentation")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Model Type:**", type(model_info['model']).__name__)
        st.write("**Use Case:**", model_info.get('use_case', 'Not specified'))
        st.write("**Bank:**", model_info.get('bank_name', 'Not specified'))
        st.write("**Target Variable:**", model_info['target_name'])
    
    with col2:
        st.write("**Features Count:**", len(model_info['feature_names']))
        st.write("**Training Date:**", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        st.write("**Model Version:**", "v1.0")
        st.write("**Risk Category:**", model_info.get('use_case', 'General'))
    
    # Fair lending/anti-discrimination analysis
    st.subheader("Fairness & Anti-Discrimination Analysis")
    
    # Simulate fairness analysis
    fairness_metrics = {
        'Demographic Parity': np.random.uniform(0.8, 1.0),
        'Equal Opportunity': np.random.uniform(0.75, 0.95),
        'Predictive Parity': np.random.uniform(0.85, 0.98)
    }
    
    col1, col2, col3 = st.columns(3)
    
    for i, (metric, value) in enumerate(fairness_metrics.items()):
        if i == 0:
            with col1:
                st.metric(metric, f"{value:.3f}", f"{'✓' if value > 0.9 else '⚠'}")
        elif i == 1:
            with col2:
                st.metric(metric, f"{value:.3f}", f"{'✓' if value > 0.9 else '⚠'}")
        else:
            with col3:
                st.metric(metric, f"{value:.3f}", f"{'✓' if value > 0.9 else '⚠'}")
    
    # Model explainability compliance
    st.subheader("Explainability Compliance")
    explainability_checks = {
        "SHAP Analysis Available": True,
        "Feature Importance Documented": True,
        "Counterfactual Explanations": False,
        "Global Explanations": True,
        "Local Explanations": True
    }
    
    for check, passed in explainability_checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        st.write(f"{status} {check}")
    
    # Risk assessment
    st.subheader("Risk Assessment")
    risk_levels = {
        'Model Risk': np.random.uniform(0.3, 0.7),
        'Data Drift Risk': np.random.uniform(0.2, 0.6),
        'Concept Drift Risk': np.random.uniform(0.1, 0.5),
        'Adversarial Risk': np.random.uniform(0.1, 0.4)
    }
    
    for risk, score in risk_levels.items():
        level = "LOW" if score < 0.4 else "MEDIUM" if score < 0.7 else "HIGH"
        color = "🟢" if score < 0.4 else "🟡" if score < 0.7 else "🔴"
        st.write(f"{color} {risk}: {score:.2f} ({level})")
    
    # Compliance status
    st.subheader("Overall Compliance Status")
    all_passed = all(explainability_checks.values()) and all(v > 0.7 for v in fairness_metrics.values())
    
    if all_passed:
        st.success("✅ Model is compliant with regulatory requirements")
        compliance_status = "APPROVED"
    else:
        st.warning("⚠️ Model requires further review before approval")
        compliance_status = "UNDER_REVIEW"
    
    # Generate compliance report
    doc_data = {
        "model_name": model_name,
        "model_type": type(model_info['model']).__name__,
        "use_case": model_info.get('use_case', 'Not specified'),
        "bank": model_info.get('bank_name', 'Not specified'),
        "features": model_info['feature_names'],
        "target_variable": model_info['target_name'],
        "fairness_metrics": fairness_metrics,
        "explainability_checks": explainability_checks,
        "risk_assessment": risk_levels,
        "overall_status": compliance_status,
        "analysis_date": datetime.now().isoformat()
    }
    
    st.download_button(
        label="Download Full Compliance Report",
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
        ["Dashboard", "Model Testing", "Model Explainability", "Internal Compliance"]
    )
    
    if page == "Dashboard":
        show_bank_dashboard()
    elif page == "Model Testing":
        show_model_testing()
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
        ["Supervisor Dashboard", "Model Comparison", "Cross-Bank Analysis", "Regulatory Compliance", "Risk Assessment", "Model Testing Results"]
    )
    
    if page == "Supervisor Dashboard":
        show_supervisor_dashboard_v2()
    elif page == "Model Comparison":
        show_cross_bank_comparison()
    elif page == "Cross-Bank Analysis":
        show_cross_bank_analysis()
    elif page == "Regulatory Compliance":
        show_regulatory_compliance()
    elif page == "Risk Assessment":
        show_risk_assessment()
    elif page == "Model Testing Results":
        show_model_testing_results()

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

def show_supervisor_dashboard_v2():
    st.subheader("Supervisor Dashboard - Model Oversight")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Banks Monitored", value="12", delta="+2")
    with col2:
        st.metric(label="Models Registered", value="45", delta="+5")
    with col3:
        st.metric(label="Models in Review", value="8", delta="-1")
    with col4:
        st.metric(label="Compliance Rate", value="87%", delta="+3%")
    
    # Model type distribution
    st.subheader("Model Distribution by Use Case")
    model_dist = pd.DataFrame({
        'Use Case': ['Credit Risk', 'Fraud Detection', 'AML', 'Market Risk', 'Operational Risk'],
        'Count': [15, 12, 8, 6, 4]
    })
    
    fig = px.pie(model_dist, values='Count', names='Use Case', title="Models by Use Case")
    st.plotly_chart(fig, use_container_width=True)
    
    # Compliance status by bank
    st.subheader("Compliance Status by Bank")
    compliance_data = pd.DataFrame({
        'Bank': ['Bank A', 'Bank B', 'Bank C', 'Bank D', 'Bank E'],
        'Compliant': [2, 4, 3, 1, 2],
        'Under Review': [1, 0, 2, 1, 0],
        'Non-Compliant': [0, 1, 0, 1, 0]
    })
    
    fig = go.Figure(data=[
        go.Bar(name='Compliant', x=compliance_data['Bank'], y=compliance_data['Compliant'], marker_color='green'),
        go.Bar(name='Under Review', x=compliance_data['Bank'], y=compliance_data['Under Review'], marker_color='orange'),
        go.Bar(name='Non-Compliant', x=compliance_data['Bank'], y=compliance_data['Non-Compliant'], marker_color='red')
    ])
    fig.update_layout(title="Model Compliance Status by Bank", barmode='stack')
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Recent Activities")
    st.write("- Bank A: Submitted credit risk model v2.1 for review")
    st.write("- Bank C: Updated fraud detection model with new features")
    st.write("- Bank E: Failed compliance check for AML model")
    st.write("- Bank B: Resubmitted model after corrections")

def show_regulatory_compliance_demo():
    st.header("Regulatory Compliance Dashboard")
    
    st.info("This view shows compliance status of models across banks according to regulatory standards.")
    
    # Mock compliance data
    banks = ["Bank A", "Bank B", "Bank C", "Bank D", "Bank E"]
    compliance_metrics = ["Documentation", "Fairness", "Performance", "Security", "Audit Trail"]
    
    compliance_data = []
    for bank in banks:
        for metric in compliance_metrics:
            score = np.random.randint(60, 100)  # Random compliance score
            status = "PASS" if score >= 80 else "REVIEW" if score >= 70 else "FAIL"
            compliance_data.append({
                'Bank': bank,
                'Metric': metric,
                'Score': score,
                'Status': status
            })
    
    compliance_df = pd.DataFrame(compliance_data)
    
    # Show overall compliance summary
    summary = compliance_df.groupby('Bank').agg({
        'Score': 'mean',
        'Status': lambda x: sum(x == 'PASS') / len(x) * 100  # Pass rate
    }).reset_index()
    summary.rename(columns={'Score': 'Avg Score', 'Status': 'Pass Rate (%)'}, inplace=True)
    summary['Pass Rate (%)'] = summary['Pass Rate (%)'].round(1)
    summary['Avg Score'] = summary['Avg Score'].round(1)
    
    st.subheader("Compliance Summary by Bank")
    st.dataframe(summary)
    
    # Detailed compliance view
    st.subheader("Detailed Compliance View")
    selected_bank = st.selectbox("Select Bank:", banks)
    
    bank_compliance = compliance_df[compliance_df['Bank'] == selected_bank]
    
    fig = px.bar(bank_compliance, x='Metric', y='Score', 
                 color='Status',
                 color_discrete_map={'PASS': 'green', 'REVIEW': 'orange', 'FAIL': 'red'},
                 title=f"Compliance Status for {selected_bank}",
                 range_y=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

def show_risk_assessment_demo():
    st.header("Risk Assessment Dashboard")
    
    st.info("Centralized view for assessing risks across all monitored banks and models.")
    
    # Create mock risk assessment data
    banks = ["Bank A", "Bank B", "Bank C", "Bank D"]
    risk_categories = ["Credit Risk", "Market Risk", "Operational Risk", "Liquidity Risk", "Model Risk"]
    
    risk_data = []
    for bank in banks:
        for category in risk_categories:
            score = np.random.randint(1, 100)
            level = "HIGH" if score > 70 else "MEDIUM" if score > 40 else "LOW"
            risk_data.append({
                'Bank': bank,
                'Risk Category': category,
                'Score': score,
                'Level': level
            })
    
    risk_df = pd.DataFrame(risk_data)
    
    # Show risk heatmap
    st.subheader("Risk Heatmap")
    pivot_risk = risk_df.pivot(index='Bank', columns='Risk Category', values='Score')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot_risk, annot=True, fmt='.0f', cmap='RdYlGn_r', center=50, ax=ax, vmin=0, vmax=100)
    ax.set_title("Risk Assessment Heatmap (Red=High Risk, Green=Low Risk)")
    st.pyplot(fig)
    plt.clf()
    
    # Show high-risk items
    high_risk = risk_df[risk_df['Score'] > 70]
    if not high_risk.empty:
        st.subheader("⚠️ High-Risk Items (>70)")
        st.dataframe(high_risk.sort_values('Score', ascending=False))
    
    # Risk trend over time (mock)
    st.subheader("Risk Trend Simulation")
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq='M')
    bank_risk_trends = {}
    
    for bank in banks:
        # Simulate risk trend for each bank
        trend = np.random.rand(len(dates)) * 30 + 50  # Base risk around 50-80
        # Add some trend variation
        trend = np.cumsum(np.random.randn(len(dates)) * 0.5) + trend
        trend = np.clip(trend, 1, 100)  # Keep in 1-100 range
        bank_risk_trends[bank] = trend
    
    risk_trend_df = pd.DataFrame({
        'Date': dates
    })
    for bank in banks:
        risk_trend_df[bank] = bank_risk_trends[bank]
    
    risk_melted = risk_trend_df.melt(id_vars=['Date'], var_name='Bank', value_name='Risk Score')
    
    fig = px.line(risk_melted, x='Date', y='Risk Score', color='Bank', 
                  title="Risk Trend Over Time")
    st.plotly_chart(fig, use_container_width=True)

def show_model_testing_results():
    st.header("Model Testing Results")
    
    st.info("Results from testing models against central bank test sets.")
    
    if 'test_results' in st.session_state and st.session_state.test_results:
        results_df = pd.DataFrame(st.session_state.test_results)
        
        st.subheader("Testing Results Summary")
        st.dataframe(results_df)
        
        # Visualize results
        st.subheader("Performance Metrics Comparison")
        
        # Melt the dataframe for plotting
        metrics_df = results_df[['model', 'accuracy', 'precision', 'recall', 'f1_score']].melt(
            id_vars=['model'], 
            value_vars=['accuracy', 'precision', 'recall', 'f1_score'],
            var_name='metric', 
            value_name='score'
        )
        
        fig = px.bar(metrics_df, x='model', y='score', color='metric',
                     title="Model Performance Metrics",
                     labels={'score': 'Score', 'model': 'Model'})
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("No testing results available yet. Test models in the Bank View to see results here.")

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
    
    st.info("Analyze and compare models across different banks for regulatory oversight.")
    
    # Use case selection
    use_cases = ["Credit Risk", "Fraud Detection", "Anti-Money Laundering", "Market Risk", "Operational Risk"]
    selected_use_case = st.selectbox("Select Use Case to Analyze:", use_cases)
    
    # Bank selection
    all_banks = ["Bank A", "Bank B", "Bank C", "Bank D", "Bank E"]
    selected_banks = st.multiselect("Select Banks to Include:", all_banks, default=all_banks[:3])
    
    if not selected_banks:
        st.warning("Please select at least one bank to analyze.")
        return
    
    if st.button(f"Generate Demo {selected_use_case} Models for Selected Banks"):
        for i, bank in enumerate(selected_banks):
            # Generate synthetic model data based on use case
            np.random.seed(42 + i)
            n_samples = 500
            
            if selected_use_case == "Credit Risk":
                # Credit risk features
                features = ['age', 'income', 'loan_amount', 'credit_score', 'employment_years']
                X = pd.DataFrame({
                    'age': np.random.normal(40, 15, n_samples),
                    'income': np.random.lognormal(10, 0.5, n_samples),
                    'loan_amount': np.random.lognormal(9, 0.8, n_samples),
                    'credit_score': np.random.normal(650, 100, n_samples),
                    'employment_years': np.random.gamma(2, 2, n_samples)
                })
                # Create target based on risk factors
                default_prob = (
                    0.1 +
                    0.3 * (X['credit_score'] < 600) +
                    0.2 * (X['income'] < 30000) +
                    0.15 * (X['loan_amount'] / X['income'] > 0.3) +
                    0.1 * (X['employment_years'] < 2) +
                    np.random.normal(0, 0.1, n_samples)
                )
                y = (np.random.random(n_samples) < default_prob).astype(int)
                
            elif selected_use_case == "Fraud Detection":
                # Fraud detection features
                features = ['transaction_amount', 'account_age_days', 'num_transactions_day', 'merchant_risk_score', 'time_since_last_transaction']
                X = pd.DataFrame({
                    'transaction_amount': np.random.lognormal(8, 1.5, n_samples),
                    'account_age_days': np.random.exponential(365, n_samples),
                    'num_transactions_day': np.random.poisson(5, n_samples),
                    'merchant_risk_score': np.random.uniform(0, 1, n_samples),
                    'time_since_last_transaction': np.random.exponential(2, n_samples)
                })
                # Create target based on fraud indicators
                fraud_prob = (
                    0.05 +
                    0.2 * (X['transaction_amount'] > 5000) +
                    0.1 * (X['merchant_risk_score'] > 0.8) +
                    0.15 * (X['num_transactions_day'] > 10) +
                    np.random.normal(0, 0.05, n_samples)
                )
                y = (np.random.random(n_samples) < fraud_prob).astype(int)
                
            else:
                # Generic features for other use cases
                features = [f"feature_{j}" for j in range(5)]
                X = pd.DataFrame({f: np.random.randn(n_samples) for f in features})
                y = (0.3*X.iloc[:, 0] + 0.2*X.iloc[:, 1] + 0.1*np.random.randn(n_samples) > 0).astype(int)
            
            # Train a simple model
            model = LogisticRegression(max_iter=1000)
            model.fit(X, y)
            
            # Save the model
            model_name = f"{bank}_{selected_use_case.lower().replace(' ', '_')}_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            st.session_state.models[model_name] = {
                'model': model,
                'X_train': X,
                'X_test': X,
                'y_train': y,
                'y_test': y,
                'feature_names': features,
                'target_name': 'risk_outcome',
                'bank_name': bank,
                'use_case': selected_use_case
            }
        
        st.success(f"Demo {selected_use_case} models created for {len(selected_banks)} banks!")
    
    if st.session_state.models:
        # Filter models by selected use case and banks
        filtered_models = {}
        for model_name, model_info in st.session_state.models.items():
            if (model_info.get('use_case', '') == selected_use_case and 
                model_info.get('bank_name', '') in selected_banks):
                filtered_models[model_name] = model_info
        
        if filtered_models:
            # Group models by bank
            bank_models = {}
            for model_name, model_info in filtered_models.items():
                bank_name = model_info.get('bank_name', 'Unknown')
                if bank_name not in bank_models:
                    bank_models[bank_name] = []
                bank_models[bank_name].append((model_name, model_info))
            
            st.subheader(f"Models by Bank - {selected_use_case}")
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
            
            # Aggregate analysis
            st.subheader(f"Aggregate Analysis - {selected_use_case}")
            
            analysis_data = []
            for model_name, model_info in filtered_models.items():
                model = model_info['model']
                X_test = model_info['X_test']
                y_test = model_info['y_test']
                
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                
                analysis_data.append({
                    'Model': model_name,
                    'Bank': model_info['bank_name'],
                    'Use Case': model_info['use_case'],
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'Features': len(model_info['feature_names'])
                })
            
            if analysis_data:
                analysis_df = pd.DataFrame(analysis_data)
                
                # Show analysis table
                st.dataframe(analysis_df)
                
                # Visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.histogram(analysis_df, x='Accuracy', 
                                      title="Distribution of Model Accuracies",
                                      nbins=10)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.box(analysis_df, x='Bank', y='Accuracy',
                                title="Accuracy by Bank",
                                color='Bank')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Correlation between metrics
                metrics_corr = analysis_df[['Accuracy', 'Precision', 'Recall']].corr()
                fig = px.imshow(metrics_corr, 
                               title="Correlation Between Performance Metrics",
                               text_auto=True)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No {selected_use_case} models available for the selected banks. Generate demo models to see analysis.")
    else:
        st.info("No models available. Generate demo models to see analysis.")

def show_risk_assessment():
    st.header("Risk Assessment Dashboard")
    
    st.info("Centralized view for assessing risks across all monitored banks and their AI models.")
    
    # Use case selection
    use_cases = ["Credit Risk", "Fraud Detection", "Anti-Money Laundering", "Market Risk", "Operational Risk"]
    selected_use_case = st.selectbox("Select Use Case:", use_cases)
    
    # Create a comprehensive risk assessment dashboard
    banks = ["Bank A", "Bank B", "Bank C", "Bank D", "Bank E"]
    
    # Generate risk data based on use case
    risk_data = []
    for bank in banks:
        for category in use_cases:
            if category == selected_use_case:
                # Higher risk scores for the selected use case
                risk_score = np.random.randint(40, 90)
            else:
                risk_score = np.random.randint(20, 70)
                
            risk_data.append({
                'Bank': bank,
                'Use Case': category,
                'Risk Score': risk_score,
                'Risk Level': 'HIGH' if risk_score > 70 else 'MEDIUM' if risk_score > 40 else 'LOW'
            })
    
    risk_df = pd.DataFrame(risk_data)
    
    # Filter for selected use case
    filtered_risk_df = risk_df[risk_df['Use Case'] == selected_use_case]
    
    st.subheader(f"Risk Assessment - {selected_use_case}")
    
    # Create heatmap for selected use case
    pivot_risk = filtered_risk_df.pivot(index='Bank', columns='Use Case', values='Risk Score')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot_risk, annot=True, fmt='.0f', cmap='RdYlGn_r', center=50, ax=ax, cbar_kws={'label': 'Risk Score'})
    ax.set_title(f"{selected_use_case} Risk Assessment by Bank")
    st.pyplot(fig)
    plt.clf()
    
    # Risk trends over time (simulation)
    st.subheader("Risk Trend Simulation")
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq='M')
    
    risk_trends = {}
    for bank in banks:
        # Simulate risk trend for each bank with some correlation to use case
        base_risk = np.random.rand(len(dates)) * 20 + 40  # Base risk around 40-60
        # Add some trend variation
        trend = np.cumsum(np.random.randn(len(dates)) * 0.8) + base_risk
        trend = np.clip(trend, 10, 100)  # Keep in 10-100 range
        risk_trends[bank] = trend
    
    risk_trend_df = pd.DataFrame({
        'Date': dates
    })
    for bank in banks:
        risk_trend_df[bank] = risk_trends[bank]
    
    risk_melted = risk_trend_df.melt(id_vars=['Date'], var_name='Bank', value_name='Risk Score')
    
    fig = px.line(risk_melted, x='Date', y='Risk Score', color='Bank', 
                  title=f"{selected_use_case} Risk Trends Over Time")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Model-specific risk assessment
    st.subheader("Model-Specific Risk Assessment")
    
    if st.session_state.models:
        # Filter models by selected use case
        filtered_models = {k: v for k, v in st.session_state.models.items() 
                          if v.get('use_case', '') == selected_use_case}
        
        if filtered_models:
            model_risk_data = []
            for model_name, model_info in filtered_models.items():
                bank_name = model_info.get('bank_name', 'Unknown')
                
                # Calculate various risk scores for the model
                model_risk = {
                    'Model': model_name,
                    'Bank': bank_name,
                    'Overfitting Risk': np.random.uniform(0.1, 0.8),
                    'Data Drift Risk': np.random.uniform(0.1, 0.7),
                    'Adversarial Risk': np.random.uniform(0.1, 0.6),
                    'Interpretability Risk': np.random.uniform(0.1, 0.9),
                    'Overall Risk Score': np.random.uniform(0.2, 0.8)
                }
                model_risk_data.append(model_risk)
            
            if model_risk_data:
                model_risk_df = pd.DataFrame(model_risk_data)
                st.dataframe(model_risk_df.round(3))
                
                # Visualize model risk scores
                fig = px.scatter(model_risk_df, x='Bank', y='Overall Risk Score', 
                                size='Overall Risk Score', color='Bank',
                                hover_data=['Model', 'Overfitting Risk', 'Data Drift Risk'],
                                title="Model Risk Scores by Bank")
                st.plotly_chart(fig, use_container_width=True)
    
    # High-risk items
    high_risk_items = filtered_risk_df[filtered_risk_df['Risk Score'] > 70]
    if not high_risk_items.empty:
        st.subheader("⚠️ High-Risk Items (>70)")
        st.dataframe(high_risk_items.sort_values('Risk Score', ascending=False))
    
    # Risk summary
    st.subheader("Risk Summary")
    avg_risk = filtered_risk_df['Risk Score'].mean()
    max_risk = filtered_risk_df['Risk Score'].max()
    high_risk_count = len(filtered_risk_df[filtered_risk_df['Risk Score'] > 70])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Risk Score", f"{avg_risk:.1f}")
    with col2:
        st.metric("Maximum Risk Score", f"{max_risk}")
    with col3:
        st.metric("High-Risk Items", f"{high_risk_count}")

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

def show_model_testing():
    st.header("🧪 Model Testing with Central Bank Test Sets")
    
    # Predefined central bank test sets
    central_bank_datasets = {
        "Credit Risk Test Set": "demo_credit_risk.csv",
        "Fraud Detection Test Set": "demo_fraud_detection.csv",
        "Anti-Money Laundering Test Set": "demo_aml.csv",
        "Market Risk Test Set": "demo_market_risk.csv",
        "Operational Risk Test Set": "demo_operational_risk.csv"
    }
    
    # Option to select from predefined test sets or upload custom
    test_source = st.radio("Select test data source:", ["Use Central Bank Test Set", "Upload Custom Test Set"])
    
    if test_source == "Use Central Bank Test Set":
        selected_dataset = st.selectbox("Select central bank test set:", list(central_bank_datasets.keys()))
        st.info(f"Selected: {selected_dataset}")
        
        # Generate demo data based on selection
        if st.button("Load Selected Test Set"):
            if selected_dataset == "Credit Risk Test Set":
                # Generate credit risk test data
                np.random.seed(42)
                n_samples = 500
                
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
                
                df = pd.DataFrame({
                    'age': age,
                    'income': income,
                    'loan_amount': loan_amount,
                    'credit_score': credit_score,
                    'employment_years': employment_years,
                    'default': default
                })
                
                df['age'] = df['age'].round().astype(int)
                df['income'] = df['income'].round(-2)
                df['loan_amount'] = df['loan_amount'].round(-2)
                df['credit_score'] = df['credit_score'].round().astype(int)
                df['employment_years'] = df['employment_years'].round(1)
                
                st.session_state.datasets[selected_dataset.lower().replace(' ', '_') + '.csv'] = df
                st.success(f"Loaded {selected_dataset} with {len(df)} samples!")
                
            elif selected_dataset == "Fraud Detection Test Set":
                # Generate fraud detection test data
                np.random.seed(42)
                n_samples = 500
                
                transaction_amount = np.random.lognormal(8, 1.5, n_samples)
                account_age_days = np.random.exponential(365, n_samples)
                num_transactions_day = np.random.poisson(5, n_samples)
                merchant_risk_score = np.random.uniform(0, 1, n_samples)
                time_since_last_transaction = np.random.exponential(2, n_samples)
                
                # Create synthetic target (fraud: 1 = fraud, 0 = legitimate)
                fraud_prob = (
                    0.05 +
                    0.2 * (transaction_amount > 5000) +
                    0.1 * (merchant_risk_score > 0.8) +
                    0.15 * (num_transactions_day > 10) +
                    np.random.normal(0, 0.05, n_samples)
                )
                fraud = (np.random.random(n_samples) < fraud_prob).astype(int)
                
                df = pd.DataFrame({
                    'transaction_amount': transaction_amount,
                    'account_age_days': account_age_days,
                    'num_transactions_day': num_transactions_day,
                    'merchant_risk_score': merchant_risk_score,
                    'time_since_last_transaction': time_since_last_transaction,
                    'fraud': fraud
                })
                
                df['transaction_amount'] = df['transaction_amount'].round(2)
                df['account_age_days'] = df['account_age_days'].round().astype(int)
                df['num_transactions_day'] = df['num_transactions_day'].round().astype(int)
                df['merchant_risk_score'] = df['merchant_risk_score'].round(3)
                df['time_since_last_transaction'] = df['time_since_last_transaction'].round(1)
                
                st.session_state.datasets[selected_dataset.lower().replace(' ', '_') + '.csv'] = df
                st.success(f"Loaded {selected_dataset} with {len(df)} samples!")
            
            # Additional test sets would be implemented similarly...
    
    else:  # Upload Custom Test Set
        uploaded_file = st.file_uploader(
            "Upload your test dataset (CSV)",
            type=["csv"],
            help="Upload a CSV file containing your test data"
        )
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.session_state.datasets[uploaded_file.name] = df
            st.success(f"Dataset '{uploaded_file.name}' uploaded successfully!")
            st.write(f"Shape: {df.shape}")
    
    # Model selection for testing
    if st.session_state.models:
        model_name = st.selectbox("Select model to test:", list(st.session_state.models.keys()))
        
        # Select test dataset
        if st.session_state.datasets:
            test_dataset_name = st.selectbox("Select test dataset:", list(st.session_state.datasets.keys()))
            
            if st.button("Run Model Testing"):
                model_info = st.session_state.models[model_name]
                model = model_info['model']
                test_data = st.session_state.datasets[test_dataset_name]
                
                # Prepare test data
                feature_cols = [col for col in test_data.columns if col != 'default' and col != 'fraud']  # Adjust based on target column
                X_test = test_data[feature_cols]
                y_test = test_data[test_data.columns[-1]]  # Assuming target is last column
                
                # Handle categorical variables
                X_test_encoded = pd.get_dummies(X_test, drop_first=True)
                
                # Make predictions
                y_pred = model.predict(X_test_encoded)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                
                st.subheader(f"Testing Results for {model_name}")
                st.metric("Accuracy", f"{accuracy:.3f}")
                
                # Detailed metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted', zero_division=0):.3f}")
                with col2:
                    st.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted', zero_division=0):.3f}")
                with col3:
                    st.metric("F1-Score", f"{f1_score(y_test, y_pred, average='weighted', zero_division=0):.3f}")
                
                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                fig = px.imshow(cm, text_auto=True, title="Confusion Matrix")
                st.plotly_chart(fig, use_container_width=True)
                
                # Save test results
                test_results = {
                    'model': model_name,
                    'dataset': test_dataset_name,
                    'accuracy': accuracy,
                    'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                    'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                if 'test_results' not in st.session_state:
                    st.session_state.test_results = []
                st.session_state.test_results.append(test_results)
                
                st.success("Model testing completed and results saved!")

def show_model_comparison_demo():
    st.subheader("Model Comparison Demo")
    st.info("This is a demonstration of model comparison functionality.")
    
    # Simulate different models from different banks
    banks = ["Bank A", "Bank B", "Bank C"]
    use_cases = ["Credit Risk", "Fraud Detection", "AML"]
    model_versions = ["v1.0", "v1.1", "v2.0"]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_bank = st.selectbox("Select Bank:", banks)
    with col2:
        selected_use_case = st.selectbox("Select Use Case:", use_cases)
    with col3:
        selected_version = st.selectbox("Select Model Version:", model_versions)
    
    if st.button("Compare Models"):
        st.subheader(f"Comparing {selected_use_case} Models")
        
        # Generate mock comparison data
        comparison_data = []
        for i, bank in enumerate(banks):
            comparison_data.append({
                'Bank': bank,
                'Use Case': selected_use_case,
                'Version': f"v{i+1}.{i}",
                'Accuracy': round(0.75 + np.random.random() * 0.2, 3),
                'Precision': round(0.70 + np.random.random() * 0.25, 3),
                'Recall': round(0.65 + np.random.random() * 0.3, 3),
                'F1-Score': round(0.70 + np.random.random() * 0.25, 3)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display comparison table
        st.dataframe(comparison_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score']))
        
        # Visualize comparison
        metric_fig = go.Figure()
        
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
            metric_fig.add_trace(go.Scatter(
                x=comparison_df['Bank'],
                y=comparison_df[metric],
                mode='lines+markers',
                name=metric,
                hovertemplate='<b>%{fullData.name}</b><br>Banks: %{x}<br>Value: %{y}<extra></extra>'
            ))
        
        metric_fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Banks",
            yaxis_title="Score",
            height=500
        )
        
        st.plotly_chart(metric_fig, use_container_width=True)

if __name__ == "__main__":
    main()