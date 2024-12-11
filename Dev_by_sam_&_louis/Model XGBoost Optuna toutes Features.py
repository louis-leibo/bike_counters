import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer, mean_squared_error
import holidays
from datetime import datetime
import datetime
import optuna
import geopandas as gpd
from shapely.geometry import Point



#######################################################
# Reading and formatting the data and the External data
#######################################################



# Import the files
df_train = pd.read_parquet("/Users/louisleibovici/Documents/VS_Code/Bike_counters DSB Project/bike_counters/data/train.parquet")
df_test = pd.read_parquet("/Users/louisleibovici/Documents/VS_Code/Bike_counters DSB Project/bike_counters/data/final_test.parquet")
#df_train = pd.read_parquet("/Users/srazjman/Python/bike_counters/data/train.parquet")
#df_test = pd.read_parquet("/Users/srazjman/Python/bike_counters/data/final_test.parquet")



############## weather ##############

# Add external data : weather data
weather = pd.read_csv(
    "/Users/louisleibovici/Documents/VS_Code/Bike_counters DSB Project/bike_counters/external_data/weather_data.csv.gz",
    #"/Users/srazjman/Python/bike_counters/external_data/weather_data.csv.gz",
    parse_dates=["AAAAMMJJHH"],
    date_format="%Y%m%d%H",
    compression="gzip",
    sep=";",
).rename(columns={"AAAAMMJJHH": "date"})

weather = weather[
    (weather["date"] >= df_train["date"].min() - datetime.timedelta(hours=1))
    & (weather["date"] <= df_test["date"].max() + datetime.timedelta(hours=1))
]

weather_reduced = (
    weather.drop(columns=["NUM_POSTE", "NOM_USUEL", "LAT", "LON", "QDXI3S"])
    .groupby("date")
    .mean()
    .dropna(axis=1, how="all")
    .interpolate(method="linear")
)

weather_reduced = (
    weather_reduced
    .drop(columns=[
        "PSTAT", "DD", "PMER", "PMERMIN", "QNEIGETOT", "QTCHAUSSEE", "ALTI", "QDRR1", "DXY", "FXY",
        "QTNSOL", "QPMER", "DXI", "QFF", "QGLO2", "QGLO", "FF", "QHFXI3S", "QINS2", "QINS",
        "QFXI3S", "RR1", "NEIGETOT", 'HXI', 'HFXI3S', "HTN", "HTX", "HUN", "HUX", "FXI3S",
        "T10", "T20", "T50", "T100", "TNSOL", "TN50", "TCHAUSSEE", "TN", "TX"
    ])
    .dropna(axis=1, how="all")
    .loc[:, weather_reduced.nunique(dropna=True) > 1]
    .drop(columns=["QTD", "QTN", "QUN", "QUX", "QTSV", "QTX", "GLO2", "INS2", "UN", "UX"])
)

# We merge :
df_train = df_train.merge(weather_reduced, left_on="date", right_on="date", how="left")
df_test = df_test.merge(weather_reduced, left_on="date", right_on="date", how="left")


############## jour_feries ##############

# Add jour ferie External data
jour_feries = (
    pd.read_csv(
        "/Users/louisleibovici/Documents/VS_Code/Bike_counters DSB Project/bike_counters/external_data/jours_feries_metropole.csv",
        #"/Users/srazjman/Python/bike_counters/external_data/jours_feries_metropole.csv",
        date_format="%Y%m%d%H"
    )
    .drop(columns=["annee", "zone"])  # Drop unnecessary columns
)

# Convert 'date' column to datetime
jour_feries['date'] = pd.to_datetime(jour_feries['date'])

# Filter rows based on the date range of df_train and df_test
jour_feries = jour_feries[
    (jour_feries["date"] >= df_train["date"].min() - datetime.timedelta(hours=1))
    & (jour_feries["date"] <= df_test["date"].max() + datetime.timedelta(hours=1))
]


############## mouvements_sociaux ##############

#Add mouvements sociaux data :
mouvements_sociaux = (
    pd.read_csv(
        "/Users/louisleibovici/Documents/VS_Code/Bike_counters DSB Project/bike_counters/external_data/mouvements-sociaux-depuis-2002.csv",
        #"/Users/srazjman/Python/bike_counters/external_data/mouvements-sociaux-depuis-2002.csv",
        date_format="%Y%m%d%H",
        sep=";"
    )
    .drop(columns=['date_de_fin', 'Organisations syndicales', 'Métiers ciblés par le préavis',
                   'Population devant travailler ciblee par le préavis', 'Nombre de grévistes du préavis'])  # Drop unnecessary columns
)

mouvements_sociaux['Date'] = pd.to_datetime(mouvements_sociaux['Date'])

mouvements_sociaux = mouvements_sociaux[
    (mouvements_sociaux["Date"] >= df_train["date"].min() - datetime.timedelta(hours=1))
    & (mouvements_sociaux["Date"] <= df_test["date"].max() + datetime.timedelta(hours=1))
]

mouvements_sociaux = mouvements_sociaux[mouvements_sociaux['Date'] != pd.Timestamp('2021-03-08')]


#######################################################
# 1. Encoding dates
#######################################################

# Extract the date feature on different time scales :

fr_holidays = holidays.France()

def _encode_dates(X):
    X = X.copy()  # Modify a copy of X

    # Encode the date information from the DateOfDeparture columns
    X["year"] = X["date"].dt.year
    X["month"] = X["date"].dt.month
    X["day"] = X["date"].dt.day
    X["weekday"] = X["date"].dt.weekday
    X["hour"] = X["date"].dt.hour

    # Creation of a binary variable depicting if the day is a weekend
    X["is_weekend"] = np.where(X["weekday"] + 1 > 5, 1, 0)

    # Add a feature to indicate if the day is a holiday in France
    X["is_holiday"] = X["date"].apply(lambda d: 1 if d in fr_holidays else 0)

    # Add a feature to indicate if it is a jour férié in France
    X["is_jour_ferie"] = X["date"].dt.date.isin(jour_feries["date"]).astype(int)

    # Add a feature to indicate if it is a jour of "mouvement social" in France
    X["is_jour_mouvement_social"] = X["date"].dt.date.isin(mouvements_sociaux["Date"]).astype(int)

    # Add morning rush and evening rush features
    X["is_working_day"] = np.where((X["weekday"] + 1 <= 5), 1, 0)
    X["morning_rush"] = ((X["hour"].between(7, 9)) & X["is_working_day"]).astype(int)
    X["evening_rush"] = ((X["hour"].between(17, 19)) & X["is_working_day"]).astype(int)

    # Add the season feature
    def season_date(date):
        if (date > datetime.datetime(2020, 9, 21)) & (date < datetime.datetime(2020, 12, 21)):
            return 1  # Autumn
        if (date > datetime.datetime(2020, 12, 20)) & (date < datetime.datetime(2021, 3, 20)):
            return 2  # Winter
        if (date > datetime.datetime(2021, 3, 19)) & (date < datetime.datetime(2021, 6, 21)):
            return 3  # Spring
        if ((date > datetime.datetime(2021, 6, 20)) & (date < datetime.datetime(2021, 9, 22))) or \
           ((date > datetime.datetime(2020, 6, 19)) & (date < datetime.datetime(2020, 9, 22))):
            return 4  # Summer
        return 0  # Fallback if none matches

    X["season"] = X["date"].apply(season_date)

    return X

df_train = _encode_dates(df_train)
df_test = _encode_dates(df_test)



#######################################
# 2. Encode Arrondissement 
#######################################

# To add an "arrondissement" feature based on latitute ande longitude
def arrondissement(X, shapefile_path="/Users/louisleibovici/Documents/VS_Code/Bike_counters DSB Project/bike_counters/external_data/arrondissements.shp"):
#def arrondissement(X, shapefile_path="/Users/srazjman/Python/bike_counters/external_data/arrondissements.shp"):

    arrondissements = gpd.read_file(shapefile_path)

    # Create a GeoDataFrame for the input dataset
    X = X.copy()  # Work on a copy of the dataset
    X["geometry"] = X.apply(lambda row: Point(row["longitude"], row["latitude"]), axis=1)
    gdf = gpd.GeoDataFrame(X, geometry="geometry", crs=arrondissements.crs)

    # Perform a spatial join to match points to arrondissements
    merged = gpd.sjoin(gdf, arrondissements, how="left", predicate="within")

    # Extract the arrondissement code (e.g., "c_ar") and fill missing values with 21
    X["district"] = merged["c_ar"].fillna(21).astype(int)

    # Drop the geometry column (optional, if not needed further)
    X = X.drop(columns=["geometry"])

    return X

df_train = arrondissement(df_train)
df_test = arrondissement(df_test)


#######################################
# 3. Covid features (Lockdown and curfew)
#######################################

# To add covid features : one binary feature for lockdown and one binary feature for curfew periods
def covid_features(data):
    # Lockdown periods
    lockdown_periods = [
        ("2020-10-30", "2020-12-15"),
        ("2021-04-03", "2021-05-03"),
    ]

    # Binary column for lockdown
    data["is_lockdown"] = 0
    for start_date, end_date in lockdown_periods:
        data.loc[
            (data["date"] >= start_date) & (data["date"] < end_date),
            "is_lockdown"
        ] = 1

    # Curfew periods with specific time restrictions
    curfew_periods = [
        ("2020-10-17", "2020-10-30", 21, 6),  # Curfew from 9 PM to 6 AM
        ("2020-12-16", "2021-01-15", 20, 6),  # Curfew from 8 PM to 6 AM
        ("2021-01-15", "2021-03-20", 19, 6),  # Curfew from 7 PM to 6 AM
        ("2021-03-20", "2021-04-03", 18, 6),  # Curfew from 6 PM to 6 AM
        ("2021-05-03", "2021-06-09", 19, 6),  # Curfew from 7 PM to 6 AM
        ("2021-06-09", "2021-06-20", 23, 6),  # Curfew from 11 PM to 6 AM
    ]

    # Binary column for curfew
    data["is_curfew"] = 0
    for start_date, end_date, start_hour, end_hour in curfew_periods:
        data.loc[
            (data["date"] >= start_date) & (data["date"] < end_date)
            & ((data["hour"] >= start_hour) | (data["hour"] < end_hour)),
            "is_curfew"
        ] = 1

    return data

# Apply the function to your datasets
df_train = covid_features(df_train)
df_test = covid_features(df_test)





################################################################
### **Preprocessing : converting features / scaling features**
################################################################

df_train = df_train.drop(columns=["counter_id", "site_id", "counter_technical_id", "coordinates"])
df_test = df_test.drop(columns=["counter_id", "site_id", "counter_technical_id", "coordinates"])

df_train = df_train.drop(columns=['date', 'is_working_day'])
df_test = df_test.drop(columns=['date', 'is_working_day'])

# Preprocessing test :

ordinal_cols = [
    "counter_installation_date"
]

onehot_cols = [
    "counter_name",
    "site_name",
]

scale_cols = [
    "latitude",
    "longitude",
    "year",
    "month",
    "day",
    "weekday",
    "hour",
    "season",
    "district",
    "T", "TD", "DG", "U", "QU", "DHUMI40", "DHUMI80", "TSV", "VV", "WW", "GLO", "INS",
]

scaler = StandardScaler()
onehot = OneHotEncoder(sparse_output=False)
ordinal = OrdinalEncoder()


############## PIPELINE ##############
# Create the preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("scale", scaler, scale_cols),
        ("onehot", onehot, onehot_cols),
        ("ordinal", ordinal, ordinal_cols),
    ]
)

# Define the full pipeline
def create_pipeline(params):
    model = XGBRegressor(**params, random_state=42)
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    return pipeline

X_train = df_train.drop(columns=["bike_count", "log_bike_count"])
y_train = df_train["log_bike_count"]

X_test = df_test.copy()

# Split the subset into train and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)



############## OPTUNA ##############

# Define the Optuna objective function
def objective(trial):
    # Suggest hyperparameters
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-8, 10.0),
    }

    # Create pipeline with suggested parameters
    pipeline = create_pipeline(param)

    # Train the pipeline
    pipeline.fit(X_train_split, y_train_split)

    # Predict on validation set
    y_pred = pipeline.predict(X_val_split)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_val_split, y_pred))
    return rmse

# Create an Optuna study and optimize
study = optuna.create_study(direction="minimize")  # Minimize RMSE
study.optimize(objective, n_trials=50, timeout=1200)  # Adjust n_trials and timeout as needed

# Get the best parameters and score
print("Best Parameters:", study.best_params)
print("Best RMSE:", study.best_value)



# #######################################
# # Prediction and creation of csv file 
# #######################################

# Train the final model with the best parameters on the full dataset
best_params = study.best_params
final_pipeline = create_pipeline(best_params)
final_pipeline.fit(X_train, y_train)

# Predict on the test set
y_predictions = final_pipeline.predict(X_test)



print(y_predictions)

pd.DataFrame(y_predictions, columns=["log_bike_count"]).reset_index().rename(
    columns={"index": "Id"}
).to_csv("/Users/louisleibovici/Documents/VS_Code/Bike_counters DSB Project/bike_counters/predictions_XGBoost_Optuna.csv", index=False)
#pd.DataFrame(y_predictions, columns=["log_bike_count"]).reset_index().rename(
#    columns={"index": "Id"}
#).to_csv("/Users/srazjman/Python/bike_counters/predictions_XGBoost_Optuna.csv", index=False)

