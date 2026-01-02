import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time



class MyScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit_transform(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std = np.where(self.std == 0, 1, self.std)
        return (X - self.mean) / self.std

    def transform(self, X):
        return (X - self.mean) / self.std

class MyLinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.cost_history = [] 

    def fit(self, X, y, penalty=0.0):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            # Ridge Regression Gradient
            dw = (1 / n_samples) * (np.dot(X.T, (y_pred - y)) + (penalty * self.weights))
            db = (1 / n_samples) * np.sum(y_pred - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            self.cost_history.append(np.mean((y_pred - y) ** 2))

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def get_statistics(self, X, y, feature_names):
        y_pred = self.predict(X)
        residuals = y - y_pred
        n = len(y)
        p = len(self.weights)
        
        sigma_squared = np.sum(residuals ** 2) / (n - p - 1)
        covariance_matrix = sigma_squared * np.linalg.inv(np.dot(X.T, X))
        std_errors = np.sqrt(np.diag(covariance_matrix))
        
        t_stats = self.weights / std_errors
        
        return pd.DataFrame({
            'Channel': feature_names,
            'Weight (ROI)': np.round(self.weights, 4),
            'T-Statistic': np.round(t_stats, 2),
            'Significant?': [' YES' if abs(t) > 2 else ' NO' for t in t_stats]
        })



st.set_page_config(page_title="Marketing ROI Engine", layout="wide")


with st.sidebar:
    st.header("Data Configuration")
    
    # 1. DEFINE THE OPTIONS LIST
    options = ["Real Industry Data (Auto-Load)", "Upload CSV", "Stress Test (50k Rows)"]
    
    # 2. CREATE THE RADIO BUTTON USING THAT LIST
    data_source = st.radio("Choose Data Source:", options)
    
   
    st.header("Model Hyperparameters")
    lr = st.slider("Learning Rate", 0.001, 0.1, 0.01, format="%.3f")
    penalty = st.slider("Ridge Penalty (L2)", 0.0, 10.0, 0.5)

st.title(" Enterprise Marketing ROI Engine")
st.markdown("A scalable, NumPy-based Ridge Regression engine built from scratch.")


df = None

# OPTION 1: REAL DATA
if data_source == "Real Industry Data (Auto-Load)":
    url = "https://raw.githubusercontent.com/justmarkham/scikit-learn-videos/master/data/Advertising.csv"
    try:
        df = pd.read_csv(url, index_col=0)
        st.success(" Connected to Live Data Repository.")
    except:
        st.error("Could not fetch data. Check internet.")

# OPTION 2: UPLOAD
elif data_source == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

# OPTION 3: STRESS TEST
elif data_source == "Stress Test (50k Rows)":
    st.warning("Generating 50,000 rows of synthetic enterprise data...")
    np.random.seed(42)
    n_rows = 50000
    tv = np.random.normal(150, 50, n_rows)
    radio = np.random.normal(30, 10, n_rows)
    newspaper = np.random.normal(40, 20, n_rows)
    sales = (0.05 * tv) + (0.2 * radio) + np.random.normal(0, 2, n_rows)
    
    df = pd.DataFrame({'TV': tv, 'Radio': radio, 'Newspaper': newspaper, 'Sales': sales})
    st.success(f" Generated {n_rows} rows. Performance Test Ready.")


if df is not None:
    # Prepare Data
    X_raw = df[['TV', 'Radio', 'Newspaper']].values
    y = df['Sales'].values
    
    scaler = MyScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # Train
    start_time = time.time()
    model = MyLinearRegression(learning_rate=lr, iterations=3000)
    model.fit(X_scaled, y, penalty=penalty)
    end_time = time.time()
    
    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows Processed", f"{len(df):,}")
    c2.metric("Training Time", f"{end_time - start_time:.4f} sec")
    c3.metric("Converged Cost", f"{model.cost_history[-1]:.4f}")
    
    # Tabs
    tab1, tab2 = st.tabs([" Insights", "Predictor"])
    
    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write(" Feature Importance")
            st.table(model.get_statistics(X_scaled, y, ['TV', 'Radio', 'Newspaper']))
        with col2:
            st.write("Residual Analysis")
            y_pred = model.predict(X_scaled)
            residuals = y - y_pred
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            sns.histplot(residuals, kde=True, ax=ax[0], color='blue')
            ax[0].set_title("Error Distribution")
            ax[1].scatter(y_pred, residuals, alpha=0.3, color='red')
            ax[1].axhline(0, color='black', linestyle='--')
            ax[1].set_title("Residuals vs Predicted")
            st.pyplot(fig)

       
        st.write(" Model Accuracy Check (Actual vs Predicted)")
        st.caption("If the points hug the dashed line, the model is fitting the data well.")
        
        fit_fig, fit_ax = plt.subplots(figsize=(10, 5))
        fit_ax.scatter(y, y_pred, color='purple', alpha=0.6, label='Actual Data Points')
        
        # Draw the Perfect Fit Diagonal Line
        min_val = min(np.min(y), np.min(y_pred))
        max_val = max(np.max(y), np.max(y_pred))
        fit_ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction Line')
        
        fit_ax.set_xlabel('Actual Sales')
        fit_ax.set_ylabel('Predicted Sales')
        fit_ax.legend()
        st.pyplot(fit_fig)

    with tab2:
        st.write(" Real-time Simulation")
        val_tv = st.slider("TV Budget", 0, 500, 150)
        val_rad = st.slider("Radio Budget", 0, 100, 30)
        val_news = st.slider("Newspaper Budget", 0, 100, 10)
        
        input_data = scaler.transform(np.array([[val_tv, val_rad, val_news]]))
        pred = model.predict(input_data)[0]
        st.metric("Predicted Sales", f"{pred:.2f} Units")

else:
    st.info("Please select a data source from the sidebar.")