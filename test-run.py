import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Assuming 'df' is your DataFrame

# One-hot encode categorical demographic data
encoder = OneHotEncoder(sparse=False)
demographic_features = encoder.fit_transform(df[['gender', 'age', 'education', 'area']])
demographic_feature_names = encoder.get_feature_names(['gender', 'age', 'education', 'area'])
demographics_df = pd.DataFrame(demographic_features, columns=demographic_feature_names, index=df.index)

# Extract day of the week and week of the year from survey_date
df['day_of_survey'] = (df['survey_date'] - df['survey_date'].min()).dt.days + 1
df['day_of_week'] = df['survey_date'].dt.dayofweek
df['week_of_year'] = df['survey_date'].dt.isocalendar().week

# Lagged features for 'ukraine_approval'
df['prev_ukraine_approval'] = df['ukraine_approval'].shift(1).fillna(method='bfill')

# Encode 'ukraine_approval' and 'prev_ukraine_approval'
target_encoder = OneHotEncoder(sparse=False)
ukraine_approval_encoded = target_encoder.fit_transform(df[['ukraine_approval']])
prev_ukraine_approval_encoded = target_encoder.transform(df[['prev_ukraine_approval']])

# Prepare final DataFrame for model input
model_df = pd.concat([
    demographics_df,
    df[['day_of_survey', 'day_of_week', 'week_of_year']],
    pd.DataFrame(prev_ukraine_approval_encoded, columns=target_encoder.get_feature_names(['prev_ukraine_approval']), index=df.index)
], axis=1)

# Prepare target variable
target_df = pd.DataFrame(ukraine_approval_encoded, columns=target_encoder.get_feature_names(['ukraine_approval']), index=df.index)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(model_df, target_df, test_size=0.2, random_state=42)
