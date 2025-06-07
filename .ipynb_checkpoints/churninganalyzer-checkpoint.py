import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBClassifier
import joblib
import streamlit as st
import plotly.express as px
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class CustomerChurnAnalyzer:
    def __init__(self,df ):
        self.churn_df = df.copy(deep=True)
        self.model = None
        self.label_encoders = {}
    def clean_data(self):
        print("The number of rows and columns in the dataset is:", self.churn_df.shape)
        print("The top five datapoints of the dataset is:")
        print(self.churn_df.head(5))
        print("The bottom five datapoints of the dataset is:")
        print(self.churn_df.tail(5))
        print(self.churn_df.dtypes)
        churn_null = self.churn_df.isnull().sum()
        print(churn_null)
        for cols in self.churn_df.columns:
            if cols == 'Customer_Age':
                self.churn_df[cols].fillna(self.churn_df[cols].mode()[0], inplace=True)
            elif self.churn_df[cols].dtype == "object":
                self.churn_df[cols].fillna(self.churn_df[cols].mode()[0], inplace=True)
            else:
                self.churn_df[cols].fillna(self.churn_df[cols].mean(), inplace=True)
        
        self.churn_df.drop(columns=[
            'Dependent_count']
             , inplace=True)
        print("The summary statistics of the dataset:")
        print(self.churn_df.describe().round(2))
        
    def remove_outliers(self):
        sns.boxplot(self.churn_df['Total_Trans_Ct'])
        plt.title("Boxplot of Total Transaction Count")
        plt.show()
        Q1 = self.churn_df["Total_Trans_Ct"].quantile(0.25)
        Q3 = self.churn_df["Total_Trans_Ct"].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        self.churn_df = self.churn_df[self.churn_df["Total_Trans_Ct"] <= upper_bound]
        sns.boxplot(self.churn_df['Total_Trans_Ct'])
        plt.title("Total Transaction Count after removing the outliers")
        plt.show()
        sns.boxplot(self.churn_df['Total_Trans_Amt'])
        plt.title("Boxplot of Total Transaction Amount")
        plt.show()
        threshold = self.churn_df['Total_Trans_Amt'].quantile(0.98)
        self.churn_df = self.churn_df[self.churn_df['Total_Trans_Amt'] <= threshold]
        sns.boxplot(self.churn_df['Total_Trans_Amt'])
        plt.title('Filtered (Top 2% Removed)')
        plt.show()
        print("The number of datapoints after removing the outliers:",self.churn_df.shape)
    def analyze_data(self):
        # Plot attrition distribution
        plt.figure(figsize=(6, 4))
        sns.countplot(x='Attrition_Flag', data=self.churn_df)
        plt.title("Distribution of Attrition Flag")
        plt.show()
    def train_churn_model(self, save_path="xgb_churn_model.pkl"):
        if self.churn_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        print("\n=== Starting Model Training ===")
        model_df = self.churn_df.drop(columns="CLIENTNUM")
        if not hasattr(self, 'label_encoders'):
            self.label_encoders = {}
        category_columns = ['Attrition_Flag', 'Gender', 'Education_Level',
                      'Marital_Status', 'Income_Category', 'Card_Category']
        for col in category_columns:
            le = LabelEncoder()
            model_df[col] = le.fit_transform(model_df[col])
            self.label_encoders[col] = le
        # Prepare features and target
        X = model_df.iloc[:, 1:19]
        y = model_df["Attrition_Flag"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Train model
        self.model = XGBClassifier(use_label_encoder=False,eval_metric="logloss",random_state=42)
        self.model.fit(X_train,y_train)
        #Feature Importance Visualization ===
        plt.figure(figsize=(10, 6))
        feature_importance = self.model.feature_importances_
        sorted_idx = np.argsort(feature_importance)[-10:]  # Get top 10 features
            
        # Create DataFrame for visualization
        importance_df = pd.DataFrame({
        'Feature': X.columns[sorted_idx],
        'Importance': feature_importance[sorted_idx]
        }).sort_values('Importance', ascending=True)  # Sort for proper plotting
    
         # Plot horizontal bar chart
        plt.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue')
        plt.xlabel('Feature Importance Score')
        plt.title('Top 10 Predictive Features for Churn Model')
    
        # Add value labels
        for index, value in enumerate(importance_df['Importance']):
            plt.text(value, index, f'{value:.3f}', va='center')
        plt.tight_layout()
        plt.show()
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        clf_report = classification_report( y_test, y_pred,target_names=['Non-Churn', 'Churn'],digits=4)
        conf_matrix = confusion_matrix(y_test, y_pred)
        # Print results
        print(f"\nAccuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(clf_report)
        print("\nConfusion Matrix:")
        print(conf_matrix)
        # Save model and label encoders
        if save_path:
            joblib.dump({'model': self.model,'label_encoders': self.label_encoders}, save_path)
            print(f"Model and encoders saved to {save_path}")
            
        return accuracy, clf_report, conf_matrix

    
    def main(self):
        self.clean_data()
        self.remove_outliers()
        self.analyze_data()
        self.train_churn_model(save_path="xgb_churn_model.pkl") 
    
                    
                   

        
                
        
           

   
        
        

    
    

    
    
    
    
    
    
    
        
