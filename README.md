# Used Car Price Prediction

A Machine Learning project that predicts the **price of a used car** based on various vehicle characteristics and history.

## Project Overview

The goal of this project is to build a Machine Learning regression model that learns from existing used-car data and predicts the expected price of a car.

The project follows this workflow:

```text

Dataset
    ↓
Data Exploration
    ↓
Data Cleaning-->{ Check Missing Values , Check Duplicates }
    ↓
Feature Engineering
    ↓
Data Preprocessing
    ↓
Train/Test Split
    ↓
Model Training
    ↓
Model Evaluation
    ↓
Price Prediction

```

## Objective

The objective is to predict the `price_usd` of a used car using information such as:

- Manufacturing year
- Mileage
- Engine capacity
- Fuel type
- Number of previous owners
- Brand
- Transmission
- Color
- Service history
- Reported accidents
- Insurance validity

## Dataset

The dataset contains the following attributes:

| Attribute | Description |
|---|---|
| `make_year` | Manufacturing year of the car |
| `mileage_kmpl` | Car mileage in kilometers per liter |
| `engine_cc` | Engine capacity in cubic centimeters |
| `fuel_type` | Type of fuel used by the car |
| `owner_count` | Number of previous owners |
| `price_usd` | Price of the car in USD — **Target Variable** |
| `brand` | Brand/manufacturer of the car |
| `transmission` | Transmission type |
| `color` | Color of the car |
| `service_history` | Whether the car has a service history |
| `accidents_reported` | Number of reported accidents |
| `insurance_valid` | Whether the car has valid insurance |

### Features

The input features (`X`) are:

```text
make_year
mileage_kmpl
engine_cc
fuel_type
owner_count
brand
transmission
color
service_history
accidents_reported
insurance_valid
```

### Target

The target variable (`y`) is:

```text
price_usd
```

Since `price_usd` is a continuous numerical value, this is a **Regression** problem.

## Technologies Used

- **Python**
- **Pandas** – Data manipulation and analysis
- **NumPy** – Numerical operations
- **Matplotlib** – Data visualization
- **Seaborn** – Exploratory data analysis
- **Scikit-learn** – Machine Learning

## Machine Learning

The project will experiment with regression algorithms such as:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

The models will be evaluated using appropriate regression metrics and compared based on their performance.

## Evaluation Metrics

The following metrics may be used to evaluate the models:

- **MAE** — Mean Absolute Error
- **MSE** — Mean Squared Error
- **RMSE** — Root Mean Squared Error
- **R² Score** — Coefficient of Determination

## Project Structure

```text
USED-CAR-PRICE-PREDICTION/
│
├── used_car_price_dataset_extended.csv
├── index.py
├── README.md
└── requirements.txt
```

The structure may be expanded as the project develops.

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
```

Navigate to the project directory:

```bash
cd USED-CAR-PRICE-PREDICTION
```

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Running the Project

Run the Python program:

```bash
python index.py
```

## Future Improvements

Possible improvements include:

- Comparing multiple regression models
- Feature engineering
- Hyperparameter tuning
- Building a web interface for price
