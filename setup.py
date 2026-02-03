from setuptools import setup, find_packages

setup(
    name="explainability-ai-tool",
    version="0.1.0",
    author="OpenClaw AI Assistant",
    author_email="jayopenclaw@gmail.com",
    description="A comprehensive system for explaining and comparing ML/AI models used by banks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Weeris/explainability-ai-tool",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial Services",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "streamlit>=1.28.0",
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "shap>=0.42.0",
        "lime>=0.2.0.1",
        "plotly>=5.15.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "requests>=2.31.0",
        "pydantic>=2.0.0",
        "sqlalchemy>=2.0.0",
        "bcrypt>=4.0.0",
        "cryptography>=41.0.0",
    ],
    entry_points={
        "console_scripts": [
            "explainability-ai-tool=app:main",
        ],
    },
)