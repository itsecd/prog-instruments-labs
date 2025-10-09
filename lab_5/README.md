# Ultimate Dataset Visualizer

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.33-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A **professional, interactive dataset visualization and analysis tool** built with **Streamlit** and **Plotly**.  
Designed for **data professionals, analysts, and researchers** to explore, analyze, and summarize CSV datasets efficiently.

---
## The App

Try the interactive dataset visualizer online: [![Open in Streamlit](https://img.shields.io/badge/Open%20in-Streamlit-orange?logo=streamlit)](https://brainytweaks-dataset-visualizer-app-2exhks.streamlit.app)

## Features

### Sleek Dark Mode UI
- Minimalistic, professional dashboard layout
- Dark mode only for a clean and focus-friendly interface

### Dataset Overview
- Total entries and fields
- Duplicate detection
- Field type distribution
- Missing values overview and heatmap

### Numeric Analysis
- Mean, Standard Deviation, Quartiles
- Skewness & Kurtosis
- Outlier detection
- Histograms, Boxplots
- Correlation heatmaps

### Categorical Analysis
- Unique value counts
- Most frequent values
- Bar charts for top categories

### Datetime Analysis
- Range and missing values
- Trend line plots

### Multi-Field Comparison
- Side-by-side numeric and categorical summaries
- Comparison plots
- Grouped analysis by categorical fields

### Export
- Download professional summary CSV

---

## Installation

1. **Clone the repository**:

```bash
git clone https://github.com/YOUR_USERNAME/dataset-visualizer.git
cd dataset-visualizer
```
2. Create a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Run the app:
```bash
streamlit run app.py
```
---

## Usage

1. Upload your CSV dataset using the sidebar.
2. Navigate through the tabs:
     - Overview
     - Numeric Analysis
     - Categorical Analysis
     - Datetime Analysis
     - Multi-Field Comparison
3. Generate interactive charts and professional summaries.
4. Export the summary CSV for reporting or presentations.

---

## Technologies

1. Streamlit - Interactive web interface
2. Pandas – Data manipulation and analysis
3. Plotly – Interactive charts
4. Python 3.12

---

## Contributing

### Contributions are welcome!

1. Fork the repository
2. Create a branch: git checkout -b feature-name
3. Commit your changes: git commit -m 'Add new feature'
4. Push to the branch: git push origin feature-name
5. Open a Pull Request

---

## License

This project is open-source under the **MIT License**.

---

> Designed to help data professionals explore datasets efficiently with a sleek, interactive, and highly functional UI.
