from numpy import NAN
import math
# from pysus.online_data.Infodengue import search_string, download
import os
import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# from wikipedia_scrap import get_metropolitan_cities

YYYYWWSTART = 201301
YYYYWWFINISH = 202401

# TODO: juntar 4 colunas de semana em 1 de mês (comprimir por mes)

# Cidades: regiao metropolina de POA
# TODO: Plot temporal de casos por mês por cidade
# TODO: Regressão linear tendo como target o numero de casos e usando umidade, temperatura etc como features

# CITIES = get_metropolitan_cities()
USELESS_COLS = ["casos_est_min", "casos_est_max", "p_rt1", "p_inc100k", "Localidade_id", "id", "versao_modelo", "tweet", "Rt", "nivel_inc", "casprov_est", "casprov_est_min", "casprov_est_max", "casconf", "notif_accum_year", "casprov"]
TEMP_UMID_COLS = ["tempmed", "tempmax", "tempmin", "umidmed", "umidmin", "umidmax"]



def create_temporal_series(csv_path):
    data = pd.read_csv(csv_path)
    X = data[["temperatura","umidade","pop"]]
    Y = data["casos"]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')

    # Display the coefficients
    coefficients = pd.DataFrame(model.coef_, X.columns,columns=['Coefficient'])
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
    city_name = clean_city_name(city_name)
    if (os.path.exists(f"./data/{city_name}.csv")):
        print(f"City {city_name} already exists.")
        return

    df = download('dengue', YYYYWWSTART, YYYYWWFINISH, city_name)
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

# for city in CITIES:
#   get_city_data(city)




if __name__ == "__main__":

    # combined_csv = concatenate_csv_files('data')
    # combined_csv.to_csv('combined.csv', index=False)
    # combined_csv = pd.read_csv("combined.csv")
    # print(combined_csv.isnull().sum())
    create_temporal_series("combined.csv")
