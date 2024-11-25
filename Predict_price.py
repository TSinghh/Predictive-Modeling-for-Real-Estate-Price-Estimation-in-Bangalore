import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pickle

# Step 1: Load Dataset
df = pd.read_csv(r"/content/Bengaluru_HR_new.csv")

# Step 2: Drop unnecessary columns
df = df.drop(['area_type', 'society', 'balcony', 'availability'], axis='columns')

# Step 3: Handle missing values
df = df.dropna()

# Step 4: Feature Engineering
# Extract BHK from size column
df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]))

# Convert total_sqft to numerical values
def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1])) / 2
    try:
        return float(x)
    except:
        return None

df['total_sqft'] = df['total_sqft'].apply(convert_sqft_to_num)
df = df.dropna(subset=['total_sqft'])

# Calculate price per sqft
df['price_per_sqft'] = df['price'] * 100000 / df['total_sqft']

# Clean up location column
df['location'] = df['location'].apply(lambda x: x.strip())
location_stats = df['location'].value_counts(ascending=False)
location_stats_less_than_10 = location_stats[location_stats <= 10]
df['location'] = df['location'].apply(lambda x: 'other' if x in location_stats_less_than_10 else x)

# Remove outliers
# Remove properties with total_sqft per BHK less than 300
df = df[~(df['total_sqft'] / df['bhk'] < 300)]

# Remove outliers based on price_per_sqft
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf['price_per_sqft'])
        st = np.std(subdf['price_per_sqft'])
        reduced_df = subdf[(subdf['price_per_sqft'] > (m - st)) & (subdf['price_per_sqft'] <= (m + st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out

df = remove_pps_outliers(df)

# Remove properties where number of bathrooms > BHK + 2
df = df[df['bath'] < df['bhk'] + 2]

# Drop unnecessary columns
df = df.drop(['size', 'price_per_sqft'], axis='columns')

# One-hot encode location
dummies = pd.get_dummies(df['location'])
df = pd.concat([df, dummies.drop('other', axis='columns')], axis='columns')
df = df.drop('location', axis='columns')

# Step 5: Define features and target variable
X = df.drop(['price'], axis='columns')
y = df['price']

# Convert any month or categorical column to numerical values
if 'month_column_name' in X.columns:
    X['month_column_name'] = X['month_column_name'].map({
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    })

# Handle other non-numeric columns
non_numeric_columns = X.select_dtypes(include=['object']).columns
for col in non_numeric_columns:
    if col == 'date_column':  # Replace with the actual column name
        X[col] = pd.to_datetime(X[col], format='%B', errors='coerce').dt.month
    else:
        X = pd.get_dummies(X, columns=[col], drop_first=True)

# Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Step 7: Model Training
lr_clf = LinearRegression()
lr_clf.fit(X_train, y_train)

# Step 8: Evaluate the model
print("Linear Regression Test Score:", lr_clf.score(X_test, y_test))

# Step 9: Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
lr_clf.fit(X_scaled, y)

# Define predict_price function
def predict_price(location, sqft, bath, bhk, proximity_to_transport, neighbourhood_rating):
    # Create a zero array with the size of input features
    x = np.zeros(len(X.columns))

    # Map inputs to respective columns
    if location in dummies.columns:
        loc_index = np.where(X.columns == location)[0][0]
        x[loc_index] = 1  # Set location to 1 for one-hot encoding

    # Set numeric inputs to respective indices
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    x[3] = proximity_to_transport
    x[4] = neighbourhood_rating

    # Standardize input features before prediction (if required)
    x_scaled = scaler.transform([x])

    # Predict the price
    return lr_clf.predict(x_scaled)[0]

# Main program
print("House Price Prediction System")
location, sqft, bath, bhk, proximity_to_transport, neighbourhood_rating = get_user_input()

# Predict the price
predicted_price = predict_price(location, sqft, bath, bhk, proximity_to_transport, neighbourhood_rating)
print(f"Predicted Price: ₹{predicted_price:.2f} lakhs")  # Price is in lakhs
