import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import xgboost as xgb
import joblib
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# Set up the page
st.set_page_config(page_title="Attrition Analysis Dashboard", layout="wide")
st.title("📊 Customer Attrition Analysis & Prediction")

# Initialize variables
churn_df = pd.DataFrame() 
model = None
label_encoders = {}

# Try to load pre-trained model automatically
try:
    model_data = joblib.load("xgb_churn_model.pkl")
    model = model_data['model']
    label_encoders = model_data.get('label_encoders', {})
    st.sidebar.success("Pre-trained model loaded successfully!")
except Exception as e:
    st.sidebar.warning(f"No model found or error loading: {str(e)}")

# Model Loading Section
st.sidebar.header("Model Options")
if st.sidebar.button("🔄 Reload Model"):
    with st.spinner('Reloading model...'):
        try:
            model_data = joblib.load("xgb_churn_model.pkl")
            model = model_data['model']
            label_encoders = model_data.get('label_encoders', {})
            st.sidebar.success("Model reloaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error reloading model: {str(e)}")

# Data Upload Section
st.sidebar.header("Data Options")
fl = st.sidebar.file_uploader("Upload customer data:", type=(["csv","txt","xlsx","xls"]))
if fl is not None:
    try:
        filename = fl.name
        churn_df = pd.read_csv(fl)
        churn_df['Attrition_Flag'] = churn_df['Attrition_Flag'].str.strip()
        st.sidebar.success("Data loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error reading file: {str(e)}")
else:
    st.sidebar.warning("Please upload a data file")

# Check if required columns exist
if not churn_df.empty:
    required_columns = ['Attrition_Flag', 'Gender', 'Marital_Status', 
                      'Education_Level', 'Income_Category', 'CLIENTNUM']
    if not all(col in churn_df.columns for col in required_columns):
        st.error("Error: Uploaded file is missing required columns for analysis")
        st.stop()

# Sidebar filters
if not churn_df.empty:
    st.sidebar.header("Filters")
    selected_gender = st.sidebar.selectbox("Gender", ["All"] + list(churn_df['Gender'].unique()))
    selected_marital = st.sidebar.selectbox("Marital Status", ["All"] + list(churn_df['Marital_Status'].unique()))
    selected_education = st.sidebar.selectbox("Education Level", ["All"] + list(churn_df['Education_Level'].unique()))
    selected_income = st.sidebar.selectbox("Income Category", ["All"] + list(churn_df['Income_Category'].unique()))

    # Apply filters
    filtered_df = churn_df.copy()
    if selected_gender != "All":
        filtered_df = filtered_df[filtered_df['Gender'] == selected_gender]
    if selected_marital != "All":
        filtered_df = filtered_df[filtered_df['Marital_Status'] == selected_marital]
    if selected_education != "All":
        filtered_df = filtered_df[filtered_df['Education_Level'] == selected_education]
    if selected_income != "All":
        filtered_df = filtered_df[filtered_df['Income_Category'] == selected_income]

# Visualization settings
sns.set_style("whitegrid")  
palette = {'Existing': '#4CAF50', 'Attrited': '#F44336'}
plotly_layout = {
    "plot_bgcolor": "white",
    "paper_bgcolor": "white",
    "margin": {"l": 50, "r": 50, "t": 80, "b": 50},
    "xaxis": {
        "showline": True,
        "linecolor": "#444",
        "linewidth": 1,
        "mirror": True,
        "showgrid": False
    },
    "yaxis": {
        "showline": True,
        "linecolor": "#444",
        "linewidth": 1,
        "mirror": True,
        "showgrid": False
    }
}

# Only show visualizations if data is loaded
if not churn_df.empty:
    # Attrition Analysis Section
    st.write("---")
    st.subheader("📊 Customer Attrition Analysis")
    
    # Row 1: Gender and Marital Status
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### Gender Distribution")
        fig1 = px.histogram(filtered_df, x='Gender', color='Attrition_Flag',
                          barmode='group',
                          color_discrete_map=palette)
        fig1.update_layout(
            title={"text": "Customer Count by Gender", "x": 0.5},
            **plotly_layout
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.write("#### Marital Status Distribution")
        fig2 = px.histogram(filtered_df, x='Marital_Status', color='Attrition_Flag',
                          barmode='group',
                          color_discrete_map=palette)
        fig2.update_layout(
            title={"text": "Customer Count by Marital Status", "x": 0.5},
            **plotly_layout
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Row 2: Education and Income
    col3, col4 = st.columns(2)
    
    with col3:
        st.write("#### Education Level Distribution")
        fig3 = px.histogram(filtered_df, x='Education_Level', color='Attrition_Flag',
                          barmode='group',
                          color_discrete_map=palette,
                          category_orders={"Education_Level": sorted(filtered_df['Education_Level'].unique())})
        fig3.update_layout(
            title={"text": "Customer Count by Education Level", "x": 0.5},
            **plotly_layout
        )
        fig3.update_xaxes(tickangle=45)
        st.plotly_chart(fig3, use_container_width=True)
    
    with col4:
        st.write("#### Income Category Distribution")
        fig4 = px.histogram(filtered_df, x='Income_Category', color='Attrition_Flag',
                          barmode='group',
                          color_discrete_map=palette,
                          category_orders={"Income_Category": sorted(filtered_df['Income_Category'].unique())})
        fig4.update_layout(
            title={"text": "Customer Count by Income Level", "x": 0.5},
            **plotly_layout
        )
        fig4.update_xaxes(tickangle=45)
        st.plotly_chart(fig4, use_container_width=True)

    # Attrition Metrics Section
    st.write("---")
    st.subheader("📌 Attrition Metrics")
    if not filtered_df.empty:
        attrition_rate = (filtered_df['Attrition_Flag'] == 'Attrited').mean()
        st.metric("Overall Attrition Rate", f"{attrition_rate:.1%}")
        st.write("Breakdown:")
        st.write(filtered_df['Attrition_Flag'].value_counts())
    else:
        st.warning("No data available for calculation")
    
    # Prediction Section
    if model is not None:
        st.write("---")
        st.subheader("🔮 Churn Prediction")
        
        if st.button("Predict High-Risk Existing Customers"):
            with st.spinner('Identifying high-risk customers...'):
                try:
                    existing_customers = filtered_df[filtered_df['Attrition_Flag'] == 'Existing'].copy()
                    
                    if len(existing_customers) == 0:
                        st.warning("No existing customers found in the filtered data")
                        st.write("Unique values in Attrition_Flag:", filtered_df['Attrition_Flag'].unique())
                        st.stop()

                    # Store original categorical columns before encoding
                    display_cols = ['CLIENTNUM', 'Customer_Age', 'Gender', 
                                    'Education_Level', 'Marital_Status', 
                                    'Income_Category', 'Card_Category']
                    display_df = existing_customers[display_cols].copy()

                    # Encode for prediction
                    for col in ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']:
                        if col in label_encoders:
                            existing_customers[col] = existing_customers[col].apply(
                                lambda x: label_encoders[col].transform([x])[0] 
                                if x in label_encoders[col].classes_ 
                                else len(label_encoders[col].classes_))

                    required_features = [
                        'Customer_Age', 'Gender', 'Education_Level', 'Marital_Status',
                        'Income_Category', 'Card_Category', 'Months_on_Book',
                        'Total_Relationship_Count', 'Months_Inactive_12_mon',
                        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio'
                    ]
                    
                    missing_features = [f for f in required_features if f not in existing_customers.columns]
                    if missing_features:
                        st.warning(f"Warning: Missing features {missing_features} - using default values")
                        for f in missing_features:
                            existing_customers[f] = 0
                    
                    X = existing_customers[required_features]
                    churn_proba = model.predict_proba(X)[:, 1]
                    display_df['Churn_Probability'] = churn_proba
                    
                    HIGH_RISK_THRESHOLD = 0.95
                    high_risk = display_df[display_df['Churn_Probability'] >= HIGH_RISK_THRESHOLD]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Existing Customers", len(display_df))
                    with col2:
                        st.metric("High-Risk Customers", len(high_risk))
                    with col3:
                        if len(display_df) > 0:
                            st.metric("High-Risk Percentage", 
                                    f"{(len(high_risk)/len(display_df)):.1%}")
                    
                    st.write("### Top 10 High-Risk Customers")
                    if len(high_risk) > 0:
                        st.dataframe(
                            high_risk.nlargest(10, 'Churn_Probability')
                            .style.format({'Churn_Probability': '{:.3%}'})
                            .background_gradient(subset=['Churn_Probability'], cmap='Reds')
                        )
                    else:
                        st.info("No customers above the high-risk threshold")

                    st.write("### Churn Probability Distribution (Existing Customers)")
                    fig_pred = px.histogram(
                        display_df, 
                        x='Churn_Probability', 
                        nbins=20,
                        color_discrete_sequence=['#FF6B6B'],
                        title="Distribution of Predicted Churn Probabilities",
                        labels={'Churn_Probability': 'Churn Probability'}
                    )
                    fig_pred.update_layout(
                        title={"text": "Churn Risk Distribution", "x": 0.5},
                        xaxis_title="Churn Probability",
                        yaxis_title="Number of Customers",
                        **plotly_layout
                    )
                    st.plotly_chart(fig_pred, use_container_width=True)

                    st.download_button(
                        label="📥 Download High-Risk Customers",
                        data=high_risk.to_csv(index=False),
                        file_name='high_risk_customers.csv',
                        mime='text/csv'
                    )

                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
    else:
        st.warning("No model loaded - predictions unavailable")

else:
    st.info("Please upload customer data to begin analysis")

