from numpy import NAN
import math
# from pysus.online_data.Infodengue import search_string, download
import os
import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from wikipedia_scrap import get_metropolitan_cities
from matplotlib.dates import DateFormatter
import seaborn as sns
import matplotlib.pyplot as plt

YYYYWWSTART = 201301
YYYYWWFINISH = 202401

CITIES = get_metropolitan_cities()
USELESS_COLS = ["casos_est_min", "casos_est_max", "p_rt1", "p_inc100k", "Localidade_id", "id", "versao_modelo", "tweet", "Rt", "nivel_inc", "casprov_est", "casprov_est_min", "casprov_est_max", "casconf", "notif_accum_year", "casprov"]
TEMP_UMID_COLS = ["tempmed", "tempmax", "tempmin", "umidmed", "umidmin", "umidmax"]

def create_temporal_series(csv_path):
    # Load data
    data = pd.read_csv(csv_path)

    # Data Exploration (Optional)
    print(data.describe())
    # sns.pairplot(data)
    # plt.show()

    # Feature and target extraction
    X = data[["temperatura", "umidade", "pop"]]
    Y = data["casos"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.1, random_state=42)

    # Model training and tuning
    model = LinearRegression()
    params = {'fit_intercept': [True, False], 'copy_X': [True, False], 'n_jobs': [None, -1, 1]}

    grid_search = GridSearchCV(model, param_grid=params, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')

    # Coefficients
    coefficients = pd.DataFrame(best_model.coef_, index=["temperatura", "umidade", "pop"], columns=['Coefficient'])
    print(coefficients)


def clean_city_name(city_name):
  city_name = city_name.replace(" ", "_")
  city_name = city_name.replace("\303\241", "a")
  city_name = city_name.replace("\303\243", "a")
  city_name = city_name.replace("\303\255", "i")
  city_name = city_name.replace("\303\251", "e")
  return city_name


def fix_umidade(df):
    umid_med = df['umidmed']
    umid_min = df['umidmin']
    umid_max = df['umidmax']

    umid_correct = []

    last_correct = 0
    for (umed, umin, umax) in zip(umid_med, umid_min, umid_max):
        if not math.isnan(umed):
            umid_correct.append(umed)
            last_correct  = umed
        else:
            tmin_not_nan = not math.isnan(umin)
            tmax_not_nan = not math.isnan(umax)
            if tmin_not_nan and tmax_not_nan:
                new_med = (umin + umax) / 2
                umid_correct.append(new_med)
                last_correct = new_med
            elif tmin_not_nan:
                umid_correct.append(umin)
                lsat_correct = umin
            else:
                umid_correct.append(last_correct)

    df['umidade'] = umid_correct
    return df


def fix_temperature(df):
    temp_med = df['tempmed']
    temp_min = df['tempmin']
    temp_max = df['tempmax']

    temp_correct = []

    last_correct = 0
    for (tmed, tmin, tmax) in zip(temp_med, temp_min, temp_max):
        if not math.isnan(tmed):
            temp_correct.append(tmed)
            last_correct  = tmed
        else:
            tmin_not_nan = not math.isnan(tmin)
            tmax_not_nan = not math.isnan(tmax)
            if tmin_not_nan and tmax_not_nan:
                temp_med = (tmin + tmax) / 2
                temp_correct.append(tmed)
                last_correct = tmed
            elif tmin_not_nan:
                temp_correct.append(tmin)
                lsat_correct = tmin
            else:
                temp_correct.append(last_correct)

    df['temperatura'] = temp_correct
    return df

def get_city_data(city_name):
  search_name = city_name
  city_name = clean_city_name(city_name)
  if (os.path.exists(f"./data/{city_name}.csv")):
      print(f"City {city_name} already exists.")
      return

  search_results = search_string(search_name)

  if not search_results:
    print(f"Could not find city {city_name}!")
    return

  if len(search_results) > 1:
    print(f"Multiple cities found for {city_name}:")
    for i, result in enumerate(search_results):
      if result == search_name:
        print(f"{i}: {result} (exact match)")
        search_name = result
        break;

  df = download('dengue', YYYYWWSTART, YYYYWWFINISH, search_name)

  if df is None:
    print(f"Could not download dengue file for city {city_name}!")
    return
  df = df.transpose()
  df = df.drop(USELESS_COLS, axis=1)
  df['data_iniSE'] = df['data_iniSE'].str.slice(0, 7)
  df = collapse_to_monthly(df)
  df['city'] = city_name
  df = fix_temperature(df)
  df = fix_umidade(df)
  df = df.drop(TEMP_UMID_COLS, axis=1)

  df.to_csv(f'data/{city_name}.csv')
  print(f"Downloaded dengue file for city {city_name}!")


def collapse_to_monthly(df):
  return df.groupby('data_iniSE').mean().reset_index()

def concatenate_csv_files(directory):
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    df_list = []

    for file in csv_files:
        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path)
        df_list.append(df)

    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df.rename(columns={'Unnamed: 0': 'id', 'umidmed': 'umidade', 'tempmed': 'temperatura'}, inplace=True)
    return combined_df

# data_directory = './data'
# file_count = len([name for name in os.listdir(data_directory) if os.path.isfile(os.path.join(data_directory, name))])

# if file_count < 34:
#   for city in CITIES:
#     get_city_data(city)


if __name__ == "__main__":
  # combined_csv = concatenate_csv_files('data')
  # combined_csv.to_csv('combined.csv', index=False)
  create_temporal_series("combined.csv")
