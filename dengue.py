from pysus.online_data.Infodengue import search_string, download
import os
import pandas as pd
from wikipedia_scrap import get_metropolitan_cities

YYYYWWSTART = 201301
YYYYWWFINISH = 202401

# TODO: juntar 4 colunas de semana em 1 de mês (comprimir por mes)

# Cidades: regiao metropolina de POA
# TODO: Plot temporal de casos por mês por cidade
# TODO: Regressão linear tendo como target o numero de casos e usando umidade, temperatura etc como features

CITIES = get_metropolitan_cities()
USELESS_COLS = ["casos_est_min", "casos_est_max", "p_rt1", "p_inc100k", "Localidade_id", "id", "versao_modelo", "tweet", "Rt", "tempmin", "umidmax", "nivel_inc", "umidmin", "tempmax", "casprov_est", "casprov_est_min", "casprov_est_max", "casconf", "notif_accum_year"]

def clean_city_name(city_name):
  city_name = city_name.replace(" ", "_")
  city_name = city_name.replace("\303\241", "a")
  city_name = city_name.replace("\303\243", "a")
  city_name = city_name.replace("\303\255", "i")
  city_name = city_name.replace("\303\251", "e")
  return city_name

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
    combined_df.rename(columns={'Unnamed: 0': 'id'}, inplace=True)
    return combined_df

for city in CITIES:
  get_city_data(city)

combined_csv = concatenate_csv_files('data')
combined_csv.to_csv('combined.csv', index=False)
