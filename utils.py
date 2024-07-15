import os
import json
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

def write_temp_price_file(df, auction_type):
    file_name = f"price_result_{auction_type}_auction.csv"
    df.to_csv(os.path.join(os.getcwd(), "bin", "temp", file_name))

def get_auction_result_prices(auction_type):
    file_name = f"price_result_{auction_type}_auction.csv"
    df = pd.read_csv(os.path.join(os.getcwd(), "bin", "temp", file_name))
    df = df.set_index("Interval")
    # df.drop(columns=["Power"], inplace=True)

    return df

def export_dict_to_temp_json(dict_data, file_name):
    with open(os.path.join(os.getcwd(), "bin", "temp",f"{file_name}.json"), "w") as f:
        json.dump(dict_data, f)

def import_dict_from_temp_json(file_name):
    with open(os.path.join(os.getcwd(), "bin", "temp", f"{file_name}.json")) as f:
        data = json.load(f)

    # Convert int string keys to int type
    clean_data = {}
    for k,v in data.items():
        
        if k == "main_results":
            clean_data[k] = {
                "system_cost": v["system_cost"],
                "marginal_price": {int(kk):vv for kk, vv in v["marginal_price"].items()},
                "demand": {int(kk):vv for kk, vv in v["demand"].items()}
                }
        else:
            clean_data[k] = {}
            for g in v.keys():
                clean_data[k][g] = {}
        
                for kk, vv in data[k][g].items():
                    new_k = int(kk) if kk.isdigit() else kk
                    clean_data[k][g][new_k] = vv
            
    return clean_data


if __name__ == "__main__":
    import_dict_from_temp_json("energy_only_auction_results")
