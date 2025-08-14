# ⚖️ AI Fairness Dashboard: Auditing and Mitigating Bias

This project is an interactive Streamlit dashboard that demonstrates how to identify and mitigate gender-based bias in a machine learning model trained to predict income levels.

## 🎯 Project Objective

The goal of this project is to provide a clear, hands-on demonstration of:
1.  **Auditing:** Quantifying the bias of a standard `Logistic Regression` model using fairness metrics like Demographic Parity and Equalized Odds.
2.  **Mitigation:** Applying post-processing mitigation techniques from the `Fairlearn` library to create a fairer model.
3.  **Visualization:** Analyzing the inherent trade-off between model accuracy and fairness.

## 🛠️ Technologies Used

- **Python:** Core programming language
- **Streamlit:** To build the interactive web dashboard
- **Scikit-learn:** For data preprocessing and modeling
- **Fairlearn:** For bias auditing and mitigation algorithms (`GridSearch`)
- **Plotly:** For creating interactive data visualizations
- **Pandas:** For data manipulation

## 📊 Key Results

- The baseline model showed a significant bias, with a **Demographic Parity Difference of 0.23**.
- By applying Fairlearn's `GridSearch` with a `DemographicParity` constraint, the disparity was **reduced by over 70%** (to 0.06) with only a minor **2%** drop in overall accuracy.
- The dashboard visualizes the trade-off frontier, allowing users to see how different levels of fairness impact predictive performance.

## ⚙️ How to Run Locally

1.  Clone the repository:
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
    cd YOUR_REPOSITORY_NAME
    ```
2.  Create and activate a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
