from pysus.online_data.Infodengue import search_string, download
import os
import sklearn.linear_model.LinearRegression

YYYYWWSTART = 202301
YYYYWWFINISH = 202401

# TODO: juntar 4 colunas de semana em 1 de mês (comprimir por mes)

# Cidades: regiao metropolina de POA
# TODO: Webscraping wikipedia regiao metropolitana de POA
# TODO: Plot temporal de casos por mês por cidade
# TODO: Regressão linear tendo como target o numero de casos e usando umidade, temperatura etc como features

CITIES = ["Porto Alegre", "Gravataí"]


def get_metropolitan_cities():
    return []


def get_city_data(city_name):
    city_name = city_name.replace(" ", "_")
    if (os.path.exists(f"./data/{city_name}.csv")):
        print(f"City {city_name} already exists.")
        return

    df = download('dengue', YYYYWWSTART, YYYYWWFINISH, city_name)
    df.to_csv(f'data/{city_name}.csv')
    print(f"Downloaded dengue file for city {city_name}!")


get_city_data("Gravataí")
