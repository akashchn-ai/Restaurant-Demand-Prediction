import streamlit as st
import pandas as pd
import joblib
import time
import os

# --- 1. Dashboard Configuration ---
st.set_page_config(
    page_title="Demand AI | Case Study 120", 
    page_icon="🍔", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Custom CSS for Attractive UI ---
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    h1 {color: #0f4c81; font-family: 'Arial Black', sans-serif;}
    h2, h3 {color: #2c3e50;}
    .stButton>button {background-color: #0f4c81; color: white; border-radius: 6px; width: 100%; font-size: 18px; height: 50px; font-weight: bold;}
    .stButton>button:hover {background-color: #1a5c99; border: 2px solid #0f4c81;}
    div[data-testid="stMetricValue"] {font-size: 38px; color: #27ae60;}
    </style>
""", unsafe_allow_html=True)

# --- 3. Load ML Model & Metadata ---
@st.cache_resource
def load_assets():
    model_path = "restaurant_demand_model.pkl"
    meta_path = "restaurant_demand_metadata.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        return None, None
        
    try:
        model = joblib.load(model_path)
        metadata = joblib.load(meta_path)
        return model, metadata
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None

pipeline, meta = load_assets()

# --- Safety Check ---
if pipeline is None or meta is None:
    st.error("🚨 **Model files not found!** You must run the Jupyter Notebook first to generate the `.pkl` files before launching this app.")
    st.stop()

# --- 4. Sidebar Navigation ---
with st.sidebar:
    st.title("🍔 DemandAI System")
    st.markdown("---")
    page = st.radio("Navigation", ["🔮 Predict Demand", "📊 Model Performance", "📖 About Project"])
    st.markdown("---")
    st.markdown(f"**Engine:** `HistGradientBoosting`\n**Accuracy (R²):** `{meta['best_r2']*100:.2f}%`")
    st.success("🟢 System Online and Ready.")

# --- 5. PAGE: Predict Demand ---
if page == "🔮 Predict Demand":
    st.title("🔮 Forecast Restaurant Demand")
    st.markdown("Adjust the configuration below to simulate a restaurant profile and predict customer engagement volume (Votes).")
    
    st.markdown("### 📝 Restaurant Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_cost = st.number_input(
            "💰 Average Cost for Two", 
            min_value=meta['avg_cost_min'], 
            max_value=meta['avg_cost_max'], 
            value=meta['avg_cost_median'],
            help="Estimated cost for two people dining."
        )
        price_range = st.select_slider(
            "🏷️ Price Range Tier", 
            options=[1, 2, 3, 4], 
            value=2,
            help="1 = Budget-friendly, 4 = Luxury dining."
        )
        
    with col2:
        table_booking = st.selectbox(
            "📅 Table Booking Supported?", 
            options=meta['table_booking_options']
        )
        online_delivery = st.selectbox(
            "🛵 Online Delivery Supported?", 
            options=meta['online_delivery_options']
        )
        
    with col3:
        aggregate_rating = st.slider(
            "⭐ Expected Aggregate Rating", 
            min_value=meta['rating_min'], 
            max_value=meta['rating_max'], 
            value=3.8, 
            step=0.1,
            help="Ratings have a massive impact on predicted demand."
        )
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Prediction Button
    if st.button("🚀 Analyze & Predict Demand"):
        with st.spinner("Processing data through ML algorithm..."):
            time.sleep(0.8) # UI delay for smooth experience
            
            # Format inputs exactly as the pipeline expects
            input_df = pd.DataFrame({
                'Average Cost for two': [avg_cost],
                'Has Table booking': [table_booking],
                'Has Online delivery': [online_delivery],
                'Price range': [price_range],
                'Aggregate rating': [aggregate_rating]
            })
            
            try:
                # Generate prediction
                pred = pipeline.predict(input_df)[0]
                final_demand = max(0, int(round(pred)))
                
                st.success("✅ Analysis Complete!")
                st.markdown("---")
                
                # Display Results
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.metric(label="🎯 Predicted Total Engagements (Votes)", value=f"{final_demand} Customers")
                with res_col2:
                    st.metric(label="📉 Error Margin (MAE)", value=f"± {int(meta['best_mae'])} Customers")
                    
                st.info("💡 **AI Recommendation:** The model places extreme weight on `Aggregate Rating`. Improving service quality to boost your rating will exponentially scale your predicted demand.")
                
            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")

# --- 6. PAGE: Model Performance ---
elif page == "📊 Model Performance":
    st.title("📊 Architecture & Performance Analysis")
    st.markdown("This dashboard operates on an advanced Scikit-Learn pipeline using Target Transformation to handle skewed data perfectly.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("### 📈 Evaluation Metrics")
        st.write(f"- **Testing R² Score:** `{meta['best_r2']:.4f}`")
        st.write(f"- **Variance Explained:** `{meta['best_r2']*100:.1f}%`")
        st.write(f"- **Mean Absolute Error:** `{meta['best_mae']:.2f}` votes")
    
    with col2:
        st.info("### ⚙️ Feature Engineering")
        st.write("- **Numerical Scaling:** `StandardScaler`")
        st.write("- **Categorical Encoding:** `OneHotEncoder`")
        st.write("- **Target Transformation:** `Log1p` to correct skewness")

# --- 7. PAGE: About Project ---
elif page == "📖 About Project":
    st.title("🎓 Case Study 120: Restaurant Demand Prediction")
    st.markdown("""
    ### Project Objectives
    This system was built to provide data-driven support for restaurant management, staffing, and inventory preparation. 
    
    **Why is this important?**
    * 🧑‍🍳 **Staffing:** Ensure enough staff are scheduled during predicted high-demand profiles.
    * 🍅 **Inventory:** Reduce food wastage by estimating the volume of expected customers based on structural features.
    * 📈 **Growth Strategies:** Identify which factors (like starting online delivery or raising ratings) generate the most demand.
    
    **Technology Stack:**
    * **Language:** Python
    * **Data Processing:** Pandas, NumPy
    * **Machine Learning:** Scikit-Learn (`HistGradientBoostingRegressor`)
    * **Deployment:** Streamlit
    """)