# Explainability AI Tool

## Overview
A comprehensive system for explaining and comparing Machine Learning/AI models used by banks. This tool enables central banks to understand, compare, and supervise the AI models used by different financial institutions, ensuring transparency and regulatory compliance. The system incorporates concepts from Project Veritas by MAS (Monetary Authority of Singapore) for explainable AI validation in financial services.

## Purpose
- Provide explainability for ML/AI models used by banks
- Enable central banks to compare models across different banks
- Ensure transparency in AI-driven financial decisions
- Support regulatory oversight of AI applications in banking
- Validate models using Project Veritas-inspired framework (Explainability, Fairness, Performance, Robustness)

## Key Features
- Model explainability engine with SHAP, LIME, and Project Veritas validation
- Cross-bank comparison framework
- Regulatory compliance module
- Visualization and reporting dashboard
- Privacy-preserving analysis capabilities
- Fairness assessment tools
- Performance and robustness validation

## Technology Stack
- Python for backend processing
- Streamlit for the interactive dashboard
- SHAP and LIME for model interpretability
- Pandas and NumPy for data processing
- Plotly for interactive visualizations
- Docker for containerization
- scikit-learn for machine learning models

## Project Veritas Integration
The tool implements concepts from Project Veritas by MAS, which provides a validation framework for Explainable AI in financial services:
- **Explainability**: Understanding how models make decisions
- **Fairness**: Ensuring models don't discriminate unfairly
- **Performance**: Validating model effectiveness
- **Robustness**: Ensuring model reliability under various conditions

## Architecture
The system consists of:
1. Data ingestion layer
2. Model explainability engine
3. Project Veritas validation framework
4. Comparison framework
5. Visualization dashboard
6. Privacy-preserving computation module

## Installation

### Local Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Weeris/explainability-ai-tool.git
   cd explainability-ai-tool
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

### Using Docker
1. Build the Docker image:
   ```bash
   docker build -t explainability-ai-tool .
   ```

2. Run the container:
   ```bash
   docker run -p 8501:8501 explainability-ai-tool
   ```

## Streamlit Sharing Deployment
The application is ready for deployment on Streamlit Community Cloud. Simply connect your GitHub repository to [share.streamlit.io](https://share.streamlit.io) and the app will be automatically deployed.

### Requirements for Streamlit Sharing
- Public GitHub repository
- `requirements.txt` file with all dependencies
- `.streamlit/config.toml` for theme customization
- Entry point at `app.py`

## Usage
1. Access the application through your browser
2. Choose between Bank View or Supervisor View
3. Use the demo models or test with central bank test sets
4. Analyze your models using the explainability tools
5. Use the Project Veritas validation framework to assess your models

## Development
To run the tests:
```bash
pip install -r requirements-test.txt
python -m pytest tests/
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.