# 📈 NetElixir Revenue Forecaster — AIgnition 3.0
### 🏆 Multi-Agent AI System & Interactive Marketing Dashboard

<p align="center">
  <img src="https://img.shields.io/badge/NetElixir-AIgnition%203.0-blue?style=for-the-badge&logo=target" alt="Hackathon">
  <img src="https://img.shields.io/badge/Track-B_Submission-orange?style=for-the-badge" alt="Track B">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python" alt="Python Version">
  <img src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit" alt="UI Streamlit">
</p>

---

## 🚀 Live Production Deployment
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://netelixir-revenue-forecaster-eg8ve26tidfdgtpdfi99tc.streamlit.app)

🔥 **Launch Live Dashboard Application:** [Open Live Cockpit](https://netelixir-revenue-forecaster-eg8ve26tidfdgtpdfi99tc.streamlit.app)

---

## 🚀 Executive Summary

Welcome to the ultimate production-ready solution for **NetElixir AIgnition 3.0 — Track B**. 

This system represents an advanced **Multi-Agent Marketing Campaign Forecasting Engine**. While Track A focuses purely on the underlying data constraints, our **Track B** suite delivers the entire enterprise architecture: a deterministic feature engineering pipeline, high-precision forecasting agents, a dynamic LLM-driven synthesis module, and an interactive simulation dashboard built using custom UI layout configurations.

---

## 🎨 Interactive UI Features & Visual Effects

The user interface (`app.py`) functions as a modern marketing analytics SaaS hub, featuring:

* **📊 Live Metric Cards:** Instantly updating ROI, forecasted revenue multipliers, and click-through rates.
* **🎚️ Dynamic Spend Sliders:** Allows decision-makers to manually scale advertising budgets across Google, Meta, and Bing to see real-time revenue impact.
* **📈 Interactive Plotly Visualizations:** Beautiful, responsive line and bar charts showing the direct correlation between ad-spend shifts and revenue projections.
* **⚡ Glassmorphic Dark Mode:** Built with a custom, high-end theme designed to match NetElixir's corporate visual identity.

---

## ⚙️ Core Algorithmic Framework

The predictive engine navigates multi-channel marketing uncertainty using a structured **Bayesian Regression Framework**. Instead of interpreting campaign coefficients as static scalars, weights are tracked as continuous probability distributions that dynamically update upon observing campaign shifts.

### The Core Bayesian Equation
Cross-channel yield behavior maps straight to Bayes' Theorem:

$$P(\theta \vert{} D) = \frac{P(D \vert{} \theta) \cdot P(\theta)}{P(D)}$$

* **Diminishing Returns (Saturation Curves):** Inputs are adjusted via non-linear transforms to capture physical marketing inflection limits—ensuring marginal yields safely contract as budgets surge.
* **Prior Regularization:** Strong non-negativity priors protect the system against multi-collinearity across data sources, shielding the runtime from mathematical anomalies.

---

## 📁 Submission Assets & Project Directory

The complete, validated hackathon documentation files are bundled right within this root directory for judging accessibility:
* 📄 **[System Architecture & Flowchart](./System_Architecture_Document.pdf):** Breaks down structural decoupling, high-speed binary serialization, and execution loops.
* 📄 **[Model Evaluation & Performance Report](./Model_Performance_Report.pdf):** Validates primary metrics ($R^2$, RMSE, MAE) and mathematical core formulas.

```text
├── data/
│   ├── google_ads_campaign_stats.csv  # Historical Google Ads data
│   ├── meta_ads_campaign_stats.csv    # Historical Meta Ads data
│   └── bing_campaign_stats.csv        # Historical Bing Ads data
├── src/
│   ├── generate_features.py           # Feature engineering agent
│   ├── predict.py                     # Inference scoring agent 
│   └── processed_features.parquet     # High-speed Apache Parquet binary layer
├── pickle/
│   └── model.pkl                      # Serialized pre-trained ML model binary
├── output/
│   └── predictions.csv                # Output targeted by the automated grader
├── app.py                             # Streamlit UI Dashboard [Track B Core]
├── requirements.txt                   # Project dependency list
├── run.sh                             # Sandbox turnkey entry point script
├── System_Architecture_Document.pdf   # Decoupled system flow architecture
├── Model_Performance_Report.pdf       # Bayesian engine performance evaluation
└── README.md                          # Documentation
