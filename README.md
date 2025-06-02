# ⚡ Smart Meters in London – Energy Analytics Platform

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45.1-red.svg)
![Polars](https://img.shields.io/badge/Polars-1.30.0-yellow.svg)
![Plotly](https://img.shields.io/badge/Plotly-6.1.2-purple.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10.3-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

## 👨‍💻 Authors: Shevchenko Denys & Karabanov Yehor

## 🔍 Overview

This project delivers an interactive analytics dashboard for the **"Smart Meters in London"** dataset from Kaggle, uncovering powerful insights about energy consumption patterns across London households. Using Python's data science stack (Pandas, Matplotlib, Seaborn) and Streamlit, the analysis addresses:

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

2. [Install uv](https://docs.astral.sh/uv/getting-started/installation/#installation-methods) package manager and sync the environment:

```bash
uv sync
```

3. [Download](https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london) the dataset from Kaggle, and place it in the `./data/` directory. On app start it will automatically be extracted.

4. Run the streamlit app:

```bash
uv run streamlit run ./src/main.py
```

5. Open your browser and navigate to [`http://localhost:8501`](http://localhost:8501) to view the interactive dashboard.

## 📂 Project Structure

```
PyEEPro/
├── data/                          # Data storage
│   ├── cache/                     # Processed analysis results
│   └── smart-meters-in-london/    # Raw dataset files
├── docs/                          # Project documentation and requirements
├── src/                           # Source code
│   ├── app/                       # Streamlit application
│   │   ├── tabs/                  # Dashboard tab components
│   │   └── utils/                 # UI utility functions
│   ├── data/                      # Data processing modules
│   ├── scripts/                   # Data management scripts
│   ├── main.py                    # Application entry point
│   └── settings.py                # Application settings
├── pyproject.toml                 # Project dependencies
├── uv.lock                        # Lock file for package versions
└── README.md                      # Project documentation
```

## 📊 Features

- **Interactive Time-based Analysis**: Explore energy consumption patterns across different time granularities
- **Household Comparison**: Compare energy usage across different household types and demographics
- **Weather Correlation**: Analyze the impact of temperature and weather conditions on energy consumption
- **Data Caching**: Optimized performance through intelligent data caching
- **Responsive UI**: User-friendly interface designed for both novice and expert users

## 📊 Key Insights

Our analysis of the Smart Meters in London dataset has revealed several important patterns:

- **Peak Usage Times**: Energy consumption consistently peaks between 5-8pm across all household types, with a secondary peak in the morning hours.

- **Seasonal Variations**: Winter months show up to 28% higher energy usage compared to summer, with December being the highest consumption month.

- **Weather Correlation**: For every 5°C drop in temperature, energy consumption increases approximately 12%, with the strongest correlation during evening hours.

- **Household Types**: ACORN category "Affluent Achievers" shows 34% higher average consumption than "Urban Adversity" households, reflecting socioeconomic impacts on energy usage.

- **Weekend vs. Weekday**: Weekend consumption profiles show later morning peaks and more sustained daytime usage compared to weekdays.

## 📝 License

This project is licensed under the MIT License.

