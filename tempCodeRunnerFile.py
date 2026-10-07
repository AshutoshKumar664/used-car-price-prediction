import pandas as pd

df=pd.read_csv("used_car_price_dataset.csv")

#-===================================Data Exploration============================================>
print(df.head())
# print(df.tail())
print(df.columns)
print(df.info())
# print(df.describe())

#===================================Inspecting the data=======================================>
print("shape:",df.shape)

print("\nMissing Values:")
# print(df.isnull().sum())


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

print(df.isnull().sum())
