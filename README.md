# ⚡ Smart Meters in London – Energy Analytics Platform

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45.1-red.svg)
![Polars](https://img.shields.io/badge/Polars-1.30.0-yellow.svg)
![Plotly](https://img.shields.io/badge/Plotly-6.1.2-purple.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10.3-green.svg)
![Seaborn](https://img.shields.io/badge/Seaborn-0.13.2-orange.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

## 👨‍💻 Authors: Shevchenko Denys & Karabanov Yehor

## 🔍 Overview

This project delivers an interactive analytics dashboard for the **"Smart Meters in London"** dataset from Kaggle, uncovering powerful insights about energy consumption patterns across London households. Using Polars for high-performance data manipulation and Streamlit for interactive visualization, the analysis addresses:

- ⏱️ **Time-based consumption trends** (hourly, daily, weekly, seasonal)
- 👪 **Household consumption patterns** across different demographics
- 🌦️ **Weather impact** on energy usage behavior
- 📊 **Comparative analytics** for business intelligence

The findings are presented through an intuitive Streamlit web application with dynamic visualizations and detailed analytics.

- **Dataset**: [Smart Meters in London on Kaggle](https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london)
- **Objective**: Extract actionable insights for energy providers, policymakers, and consumers regarding electricity usage patterns in London.

## 🛠️ Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/#installation-methods) - An extremely fast Python package and project manager, written in Rust.
- Python 3.13+ recommended

## 🚀 Technologies

This project leverages powerful data science and visualization libraries to deliver high-performance analytics:

- **Polars** (≥1.30.0): A lightning-fast DataFrame library implemented in Rust, providing efficient data manipulation capabilities with a pandas-like API but significantly improved performance for large datasets.

- **Streamlit** (≥1.45.1): Creates the interactive web application with minimal code, enabling rapid development of data-driven applications with automatic hot-reloading.

- **Plotly** (≥6.1.2): Provides interactive, publication-quality graphs and visualizations with zoom, hover, and selection tools that enhance user exploration capabilities.

- **Matplotlib** (≥3.10.3): Generates high-quality static visualizations with precise control over styling and layout, used for detailed analytical charts.

- **Seaborn** (≥0.13.2): Built on Matplotlib, delivers beautiful statistical visualizations with enhanced aesthetics and simplified complex plot creation.

## 📦 Installation

1. Clone the repository:

```bash
git clone https://github.com/LilConsul/PyEEPro.git
cd PyEEPro
```

2. [Install uv](https://docs.astral.sh/uv/getting-started/installation/#installation-methods) package manager and create/sync the environment in the project directory:

```bash
uv sync
```

3. [Download](https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london) the dataset from Kaggle, and place the zip file in the `./data/` directory as `smart-meters-in-london.zip`. The application will automatically extract it on startup.

4. Run the streamlit app:

```bash
uv run streamlit run ./src/main.py
```

5. Open your browser and navigate to [`http://localhost:8501`](http://localhost:8501) to view the interactive dashboard.

## 📂 Project Structure

```
PyEEPro/
├── data/                          # Data storage
│   ├── cache/                     # Processed analysis results cached for performance
│   └── smart-meters-in-london/    # Raw dataset files (auto-extracted from .zip)
├── docs/                          # Project documentation and requirements
├── src/                           # Source code
│   ├── app/                       # Streamlit application components
│   │   ├── tabs/                  # Dashboard tab components for different analyses
│   │   └── utils/                 # UI utility functions and visualization helpers
│   ├── data/                      # Data processing and management modules
│   ├── scripts/                   # Dataset handling scripts (validation, extraction)
│   ├── main.py                    # Application entry point
│   └── settings.py                # Application settings
├── pyproject.toml                 # Project dependencies
├── uv.lock                        # Lock file for package versions
├── LICENSE                        # MIT License file
└── README.md                      # Project documentation
```

## 📊 Features and Dashboard Tabs

The application is organized into three main interactive tabs:

### 📈 Time-based Trends
- Analyze energy consumption patterns across different time periods:
  - Hourly consumption profiles
  - Daily energy usage variations 
  - Weekly consumption patterns
  - Seasonal changes in energy demand
  - Weekday vs weekend usage comparisons

### 📊 Household Behavior
- Explore how different household types consume electricity:
  - Comparison across demographic segments
  - Analysis by ACORN economic category
  - Household size and composition impact

### 📉 Weather Impact
- Understand how weather conditions affect energy usage:
  - Temperature correlation with consumption
  - Seasonal weather patterns and energy demand
  - Hourly temperature and usage relationships

## 🔧 Performance Optimization

The application uses several strategies to maintain performance with large datasets:

- **Data Caching**: Processed analysis results are cached to minimize redundant calculations
- **Polars Dataframes**: Lightning-fast data processing through Rust-powered Polars
- **Selective Data Loading**: Only the necessary data is loaded based on user selections

## 📝 License

This project is licensed under the MIT License.

