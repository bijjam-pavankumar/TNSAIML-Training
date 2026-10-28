import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, accuracy_score,mean_absolute_error
from sklearn.model_selection import train_test_split
from pathlib import Path
import joblib

 
#Step 2:Read the csv file
data=Path("house-prices.csv")
df=pd.read_csv(data)

#Step 3: EDA process
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.columns.tolist())

#Step 4: Split train and test
df.columns=df.columns.str.strip()

# Convert categorical string variables into dummy/indicator variables
df_encoded = pd.get_dummies(df, columns=['Brick', 'Neighborhood'], drop_first=True)

# Define target column as 'Price' and features excluding 'Price' and 'Home' ID column
x=df_encoded.drop(["Price", "Home"],axis=1)
y=df_encoded["Price"]

x_train,x_test,y_train,y_test=train_test_split(
    x ,y,
    test_size=0.3,
    random_state=30,
    )

#training
model=LinearRegression()
model.fit(x_train,y_train)
x_train_pred=model.predict(x_train)

# Using R-squared score for evaluating a regression model
training_r2=r2_score(y_train, x_train_pred)
print(f"Training R-squared Score: {training_r2:.4f}")

joblib.dump(model, "model.joblib")
joblib.dump(x.columns.tolist(), "feature_names.joblib")




