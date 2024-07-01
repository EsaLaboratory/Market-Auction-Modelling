import os
import pandas as pd


def get_demand_data():
    demand = pd.read_csv(os.path.join(os.getcwd(), "bin", "demand.csv")) # 48 trading periods
    demand = demand[["PowerMW"]].values.tolist()
    demand = [e[0] for e in demand]

    return demand

def get_generator_data():
    df = pd.read_csv(os.path.join(os.getcwd(), "bin", "thermal_generators.csv"))
    data_dict = df.set_index("Generator").T.to_dict("dict")

    return data_dict

def get_wind_generator_data():
    generation = pd.read_csv(os.path.join(os.getcwd(), "bin", "wind_generation.csv"))
    generation = generation.set_index("Generator")
    generation = generation.T
    generation.index = generation.index.astype(int)

    return generation

def get_solar_generator_data():
    generation = pd.read_csv(os.path.join(os.getcwd(), "bin", "solar_generation.csv"))
    generation = generation.set_index("Generator")
    generation = generation.T
    generation.index = generation.index.astype(int)

    return generation

def get_storage_data():
    df = pd.read_csv(os.path.join(os.getcwd(), "bin", "storage.csv"))
    data_dict = df.set_index("Storage").T.to_dict("dict")

    return data_dict

def write_temp_csv_file(df, file_name):
    df.to_csv(os.path.join(os.getcwd(), "bin", "temp", f"{file_name}.csv"))

def get_forecast_prices():
    df = pd.read_csv(os.path.join(os.getcwd(), "bin", "temp", "price_forecast.csv"))
    df = df.set_index("Interval")
    df.drop(columns=["Power"], inplace=True)

    return df

if __name__ == "__main__":
    df = get_forecast_prices()
    print(df)
    
