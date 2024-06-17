import requests
from bs4 import BeautifulSoup

URL = "https://pt.wikipedia.org/wiki/Regi%C3%A3o_Metropolitana_de_Porto_Alegre"

def get_metropolitan_cities():
    response = requests.get(URL)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'class': 'wikitable'})
    cities = []
    for row in table.find_all('tr')[1:]:
        city = row.find_all('td')[0].text.strip()
        if city == "Total":
          continue
        cities.append(city)

    assert len(cities) == 34, f"Expected 34 cities, got {len(cities)}"
    return cities
