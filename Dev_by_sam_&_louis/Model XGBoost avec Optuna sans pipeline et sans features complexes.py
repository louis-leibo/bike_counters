import pandas as pd
import numpy as np
import xgboost as xgb
import holidays
from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import optuna
import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import geopandas as gpd
from shapely.geometry import Point

# Import the files
df_train = pd.read_parquet("/Users/louisleibovici/Documents/VS_Code/Bike_counters DSB Project/bike_counters/data/train.parquet")
df_test = pd.read_parquet("/Users/louisleibovici/Documents/VS_Code/Bike_counters DSB Project/bike_counters/data/final_test.parquet")
# df_train = pd.read_parquet("/Users/srazjman/Python/bike_counters/data/train.parquet")
# df_test = pd.read_parquet("/Users/srazjman/Python/bike_counters//data/final_test.parquet")

#####################################################################
###ADDING NEW FEATURES AND ADDING EXTERNAL DATA TO ADD NEW FEATURES##
#####################################################################

###### JOUR FERIE ######

# https://www.data.gouv.fr/fr/datasets/jours-feries-en-france/

# Add jour ferie data
jour_feries = (
    pd.read_csv(
        "/Users/louisleibovici/Documents/VS_Code/Bike_counters DSB Project/bike_counters/external_data/jours_feries_metropole.csv",
        # "/Users/srazjman/Python/bike_counters/external_data/jours_feries_metropole.csv",
        date_format="%Y%m%d%H"
    )
    .drop(columns=["annee", "zone"])
)

#We convert 'date' column to datetime
jour_feries['date'] = pd.to_datetime(jour_feries['date'])

#We know filter rows based on the date range of df_train and df_test :
jour_feries = jour_feries[
    (jour_feries["date"] >= df_train["date"].min() - datetime.timedelta(hours=1))
    & (jour_feries["date"] <= df_test["date"].max() + datetime.timedelta(hours=1))
]

###### MOUVEMENTS SOCIAUX ######

# https://ressources.data.sncf.com/explore/dataset/mouvements-sociaux-depuis-2002/export/?sort=date_de_debut

#With the same way, we add mouvements sociaux data :
mouvements_sociaux = (
    pd.read_csv(
        "/Users/louisleibovici/Documents/VS_Code/Bike_counters DSB Project/bike_counters/external_data/mouvements-sociaux-depuis-2002.csv",
        # "/Users/srazjman/Python/bike_counters/external_data/mouvements-sociaux-depuis-2002.csv",
        date_format="%Y%m%d%H",
        sep=";"
    )
    .drop(columns=['date_de_fin', 'Organisations syndicales', 'Métiers ciblés par le préavis',
                   'Population devant travailler ciblee par le préavis', 'Nombre de grévistes du préavis'])
)

mouvements_sociaux['Date'] = pd.to_datetime(mouvements_sociaux['Date'])
mouvements_sociaux = mouvements_sociaux[
    (mouvements_sociaux["Date"] >= df_train["date"].min() - datetime.timedelta(hours=1))
    & (mouvements_sociaux["Date"] <= df_test["date"].max() + datetime.timedelta(hours=1))
]
mouvements_sociaux = mouvements_sociaux[mouvements_sociaux['Date'] != pd.Timestamp('2021-03-08')]

###### DATE FEATURES ######

# Extract the date feature on different time scales :

fr_holidays = holidays.France()

def _encode_dates(X):
    X = X.copy()

    #We first encode the date information from the DateOfDeparture columns
    X["year"] = X["date"].dt.year
    X["month"] = X["date"].dt.month
    X["day"] = X["date"].dt.day
    X["weekday"] = X["date"].dt.weekday
    X["hour"] = X["date"].dt.hour

    #Then, we create a binary variable depicting if the day is a weekend
    X["is_weekend"] = np.where(X["weekday"] + 1 > 5, 1, 0)

    #We add a feature to indicate if the day is a holiday in France
    X["is_holiday"] = X["date"].apply(lambda d: 1 if d in fr_holidays else 0)

    #Same if it is a jour férié in France
    X["is_jour_ferie"] = X["date"].dt.date.isin(jour_feries["date"]).astype(int)

    #And finally, same if it is a jour of "mouvement social" in France
    X["is_jour_mouvement_social"] = X["date"].dt.date.isin(mouvements_sociaux["Date"]).astype(int)

    return X

df_train = _encode_dates(df_train)
df_test = _encode_dates(df_test)

###### ARRONDISSEMENT FEATURE ######

# https://opendata.paris.fr/explore/dataset/arrondissements/export/?disjunctive.c_ar&disjunctive.c_arinsee&disjunctive.l_ar

# To add an "arrondissement" feature based on latitute and longitude
# def arrondissement(X, shapefile_path="/Users/srazman/Python/bike_counters/external_data/arrondissements.shp"):
def arrondissement(X, shapefile_path="/Users/louisleibovici/Documents/VS_Code/Bike_counters DSB Project/bike_counters/external_data/arrondissements.shp"):

    arrondissements = gpd.read_file(shapefile_path)

    #We create a "GeoDataFrame" for the input dataset
    X = X.copy()
    X["geometry"] = X.apply(lambda row: Point(row["longitude"], row["latitude"]), axis=1)
    gdf = gpd.GeoDataFrame(X, geometry="geometry", crs=arrondissements.crs)

    #Then, perform a spatial join to match points to arrondissements
    merged = gpd.sjoin(gdf, arrondissements, how="left", predicate="within")

    #And extract the arrondissement code
    X["district"] = merged["c_ar"].fillna(21).astype(int)
    X = X.drop(columns=["geometry"])

    return X

df_train = arrondissement(df_train)
df_test = arrondissement(df_test)

df_train = df_train.drop(columns=['date'])
df_test = df_test.drop(columns=['date'])

###### COUNTER_INSTALLATION DATE ######

# Extract features from counter_installation_date
for df in [df_train, df_test]:
    df["installation_year"] = df["counter_installation_date"].dt.year
    df["installation_month"] = df["counter_installation_date"].dt.month

df_train = df_train.drop(columns=["counter_installation_date"])
df_test = df_test.drop(columns=["counter_installation_date"])

#####################################################################
############################PREPROCESSING############################
#####################################################################

# Label encode high-cardinality categorical features
label_encoders = {}


for col in ["counter_id", "site_id", "counter_name", "site_name", "counter_technical_id", "coordinates"]:
    le = LabelEncoder()
    df_train[col] = le.fit_transform(df_train[col])
    df_test[col] = le.fit_transform(df_test[col])
    label_encoders[col] = le


X_train = df_train.drop(columns=["bike_count", "log_bike_count"])
y_train = df_train["log_bike_count"]

X_test = df_test.copy()

#We choose to split the subset into train and validation sets as follows :
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

#####################################################################
################################MODEL################################
#####################################################################

# Define the Optuna objective function
def objective(trial):
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

    model = xgb.XGBRegressor(**param, random_state=42)
    model.fit(X_train_split, y_train_split)

    y_pred = model.predict(X_val_split)

    rmse = np.sqrt(mean_squared_error(y_val_split, y_pred))
    return rmse

#We create an Optuna study and optimize
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100, timeout=1200)

#We train the final model with the best parameters on the full dataset
best_params = study.best_params
best_model = xgb.XGBRegressor(**best_params, random_state=42)
best_model.fit(X_train, y_train)

#And finally, we predict on the test set
y_predictions = best_model.predict(X_test)

pd.DataFrame(y_predictions, columns=["log_bike_count"]).reset_index().rename(
    columns={"index": "Id"}
).to_csv("/Users/louisleibovici/Documents/VS_Code/Bike_counters DSB Project/bike_counters/predictions_XGBoost_Optuna_sanspipeline.csv", index=False)