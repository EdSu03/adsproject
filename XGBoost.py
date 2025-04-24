import xgboost as xgb
import pandas as pd
import model_helper_functions as mod
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error,  r2_score
from sklearn.preprocessing import LabelEncoder

df =  pd.read_parquet("./all_cleaned_data/all_cleaned_data.parquet")

df = mod.necessary_fields(df)

df['Hour'] = df['DropoffDatetime'].apply(mod.round_time_to_int)

df = df.drop(columns = ['PickupDatetime', 'DropoffDatetime', 'TripDuration', 'TripDistance', 'FareAmount', 'TipAmount'])

X,y = df[['DOLocationID', 'Hour']], df['PULocationID']
del df

le = LabelEncoder()
y = le.fit_transform(y)
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.1, random_state=1)

model = xgb.XGBClassifier(n_estimators = 100, max_depth = 2, eta = 0.1, subsample = 0.1)
model.fit(train_X, train_y)

# Training performance:
pred_y = model.predict(train_X)

print("Train R2-score:", r2_score(train_y, pred_y))
print("Train RMSE:", root_mean_squared_error(train_y, pred_y))

# Testing performance:
pred_y = model.predict(test_X)

print("Test R2-score:", r2_score(test_y, pred_y))
print("Test RMSE:", root_mean_squared_error(test_y, pred_y))

# Saving Model:
model.save_model("./xgboost_models/xgboost_test_classifier.json")