import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.append(os.path.join(dir_path, '../rutracker'))
from qry.qry import qry
import pdb

df = qry(weight_type=None, datacachefile=True, datacachedate="2024-04-27")
pdb.set_trace()

DEMO_VARS = ['gender', 'age_range', 'education', 'area']
OUTCOME_VARS = ['ukraine_approval', 'statementagree_smo_successful']

# Subset df to only include values of survey_date where there is at least one non-null value for the outcome variables
nonnulldates = df.survey_date.loc[df[OUTCOME_VARS].notnull().any(axis=1)].unique()
df = df.loc[df.survey_date.isin(nonnulldates)]

vocab = {}
for var in DEMO_VARS:
    ctr = 1
    vocab[var] = {}
    for val in df[var].unique():
        vocab[var][val] = ctr
        ctr += 1
    df[var] = df[var].map(vocab[var])

start_date = df.survey_date.min()
df['day_of_survey'] = (df['survey_date'] - start_date).dt.days + 1
df['day_of_week'] = df['survey_date'].dt.dayofweek
df['week_of_year'] = df['survey_date'].dt.isocalendar().week

DAY_VARS = ['day_of_survey', 'day_of_week', 'week_of_year']
df = df[DEMO_VARS + DAY_VARS + OUTCOME_VARS]


# One-hot encode categorical demographic data
encoder = OneHotEncoder(sparse=False)
demographic_features = encoder.fit_transform(df[['gender', 'age_range', 'education', 'area']])
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
