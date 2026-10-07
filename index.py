import pandas as pd

df=pd.read_csv("used_car_price_dataset.csv")

#-===================================Data Exploration============================================>
print("First 5 rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
df.info()

print("\nDataset Shape:")
print(df.shape)

#===================================Inspecting the data=======================================>

print("\nMissing Values:")
print(df.isnull().sum())


print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nData Type:")
print(df.dtypes)

print("\nBasic Statistics:")
print(df.describe())

#==================================-data Cleaning============================================>

#-->handling the missing value as we inspect everything about data 

# print(df["service_history"])

print(df["service_history"].value_counts(dropna=False))  

df["service_history"]=df["service_history"].fillna("unknown")


print(df["service_history"].value_counts())
#==================================seprate feature and target variable
X=df.drop("price",axis=1)
y=df["price"]

#======================Converting Catagorical Data Into Numeric Data===============================================>

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

categorical_columns = [
    "fuel_type",
    "brand",
    "transmission",
    "color",
    "service_history",
    "insurance_valid"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(), categorical_columns)
    ],
    remainder="passthrough"
)

#=============Splitting the Data===============================================>

from sklearn.model_selection import train_test_split

x_train ,x_test ,y_train ,y_test=train_test_split(
    X, y, test_size=0.2, random_state=42
)


print("\nTraining Set Shape:")
print(x_train.shape)

print("\nTesting Set Shape:")
print(x_test.shape)

print("\nTraining Set Target Variable Shape:")
print(y_train.shape)

print("\nTesting Set Target Variable Shape:")
print(y_test.shape)

#=============================PREPROCESSING============>

X_transformed = preprocessor.fit_transform(x_train)

x_test_proceed = preprocessor.transform(x_test)

print("\nProcessed training data shape:")
print(X_transformed.shape)

print("\nProcessed testing data shape:")
print(x_test_proceed.shape)

#=============================Model Training===========>



#-----------------------Linear Regression
from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()
linear_model.fit(X_transformed, y_train)



#-----------------------decision tree regression
from sklearn.tree import DecisionTreeRegressor

decision_tree_model = DecisionTreeRegressor(random_state=42)

decision_tree_model.fit(X_transformed, y_train)


#======================predicting the model=========================>

