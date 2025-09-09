# AutoMLWeb

A beautiful and interactive Streamlit web app for automatic machine learning (AutoML) on any CSV file.  
**This project is designed for Apple Silicon (M1/M2/M3) environments.**

---

## Features

- 📊 **EDA (Exploratory Data Analysis):** Instantly preview your data, see basic statistics, and visualize column distributions.
- 🧠 **Automatic Task Detection:** The app intelligently detects whether your problem is **classification**, **regression**, or **clustering** based on your chosen target column.
- 🤖 **AutoML Model Training:** Automatically select and train the best model for your data using PyCaret.
- 🏆 **Results Visualization:** Compare model scores in sorted charts and tables for intuitive understanding.
- ✨ **Feature Selection & Retraining:** Select important features automatically and retrain models for comparison.
- 📈 **Score Comparison:** Visualize and compare the best model scores before and after feature selection.

---

## How to Use

### 1. Install dependencies

```sh
pip install -r requirements.txt
```

### 2. Run the web app

```sh
streamlit run src/app.py
```

### 3. Open the app

Visit [http://localhost:8501](http://localhost:8501) in your browser.

---

## Docker (Apple Silicon Only)

1. Build the Docker image:

   ```sh
   docker build -t automlweb .
   ```

2. Run the container (mount your code for live editing):

   ```sh
   docker run -p 8501:8501 -v $(pwd)/src:/app/src automlweb
   ```

---

## App Workflow

### 1️⃣ EDA

> **What happens:**
>
> - Preview the first few rows of your uploaded CSV.
> - View basic statistics (`df.describe()`).
> - See dynamic pie charts for the distribution of categorical columns and binned numerical columns.

### 2️⃣ Model Training (AutoML)

> **What happens:**
>
> - You select the target (label) column, or "No target" for clustering.
> - The app automatically detects the task type (Classification, Regression, or Clustering).
> - PyCaret is used to automatically train and compare multiple models.
> - The best model is selected based on the default metric for the task.
> - Results are stored for later visualization.

### 3️⃣ Results Visualization

> **What happens:**
>
> - The best model from the initial training run is displayed.
> - A detailed table shows the performance scores of all models that were compared.

### 4️⃣ Feature Selection & Retraining

> **What happens:**
>
> - For classification and regression tasks, you can run feature selection.
> - The app uses a Random Forest estimator to identify the most important features.
> - A new set of models is trained using only these selected features.
> - The new results are stored for comparison.

### 5️⃣ Score Comparison

> **What happens:**
>
> - The app compares the best model's score from the original run with the best model's score after feature selection.
> - The comparison is visualized with `st.metric` and a line chart.
> - The selected features are listed for your reference.

---

## What is AutoML?

**AutoML** (Automated Machine Learning) is a process that automatically selects, trains, and tunes machine learning models for your data.  
It saves you time and effort by automating tasks like:

- Trying multiple algorithms
- Optimizing hyperparameters
- Selecting the best model

In this app, AutoML is powered by [PyCaret](https://pycaret.gitbook.io/docs/), which makes machine learning easy and fast.

---

## Notes

- This app is optimized for Apple Silicon (M1/M2/M3) environments.
- Make sure your CSV file is clean and the label column is correctly selected.
- Feature selection may not always improve scores; results depend on your data.

---

## License

MIT
