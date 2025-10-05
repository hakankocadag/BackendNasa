import requests
import pandas as pd
from io import StringIO
from usa_cities import values
def getting_data(tup):
    city_name, lat, lon = tup
    parameters = [
        'T2M', 'T2M_MAX', 'T2M_MIN', 'RH2M', 'WS2M',
        'ALLSKY_SFC_UV_INDEX', 'CLOUD_OD', 'PRECTOTCORR',
        'PS', 'WD2M', 'ALLSKY_KT'
    ]
    parameters = ",".join(parameters)
    url = f'https://power.larc.nasa.gov/api/temporal/daily/point?parameters={parameters}&community=SB&latitude={lat}&longitude={lon}&start=20240101&end=20251005&format=CSV'
    res = requests.get(url)
    data_parts = res.text

    if '-END HEADER-' in data_parts:
        actual_data = data_parts.split('-END HEADER-')[1]

        df = pd.read_csv(StringIO(actual_data))

        df['city'] = city_name
        df['latitude'] = lat
        df['longitude'] = lon

        return df
    else:
        print(f"Failed for city {city_name}: {data_parts}")
        return None


all_dfs = []
cities = values

for city in cities:
    df_city = getting_data(city)
    if df_city is not None:
        all_dfs.append(df_city)

final_df = pd.concat(all_dfs, ignore_index=True)
final_df.to_csv("NASA/nasa_weather_data.csv", index=False)
print(final_df.head())
