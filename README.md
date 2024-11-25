# Predictive-Modeling-for-Real-Estate-Price-Estimation-in-Bangalore
The House Price Prediction System is a machine learning project designed to predict the prices of residential properties based on various features such as location, total square footage, number of bedrooms (BHK), bathrooms, proximity to transport, and neighborhood ratings. The system uses a Linear Regression model, trained on a clean and processed dataset, to provide accurate price predictions. This project demonstrates end-to-end implementation, from data preprocessing and feature engineering to model training and deployment.

# Features
# 1. Data Preprocessing:
Handled missing values and inconsistent data.
Engineered key features like BHK count and price per square foot.
Removed outliers based on property dimensions and price per square foot.

# 2. Feature Engineering:
Extracted numerical values from text columns (e.g., "size" column).
Converted location data to a one-hot encoded format for modeling.
Standardized features to optimize model performance.

# 3. Model Training:
Used a Linear Regression model for training and evaluation.
Performed train-test split to validate model accuracy.
Incorporated StandardScaler for feature standardization.

# 4. User Input Prediction:
Developed an interactive system where users input property details like location, square footage, BHK, and proximity to amenities to get a dynamic price prediction.
