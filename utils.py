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


def linearise_conventional_generator_costs():
    
    number_of_segments = 10
    generator_costs_dict = {}

    generator_dict = get_generator_data()
    
    for g in generator_dict.keys():

        a = generator_dict[g]["cost_per_mw2h"]
        b = generator_dict[g]["cost_per_mwh"]
        c = generator_dict[g]["cost_per_h"]

        p_min = generator_dict[g]["min_power_mw"]
        p_max = generator_dict[g]["max_power_mw"]
        
        base_cost_per_hour = a * p_min**2 + b * p_min + c
        segment_width = (p_max - p_min) / number_of_segments

        segments = {n:{} for n in range(1, number_of_segments + 1)}

        for n in range(1, number_of_segments + 1):
            lower_limit = p_min + (n - 1) * segment_width
            upper_limit = p_min + n * segment_width
            
            lower_limit_cost = a * lower_limit**2 + b * lower_limit + c
            upper_limit_cost = a * upper_limit**2 + b * upper_limit + c

            segment_cost = (upper_limit_cost - lower_limit_cost) / segment_width

            segments[n]["cost_per_mwh"] = segment_cost
            segments[n]["capacity_mw"] = segment_width
            segments[n]["lower_limit_mw"] = lower_limit
            segments[n]["upper_limit_mw"] = upper_limit
        
        segments["base_cost_per_hour"] = base_cost_per_hour
        
        generator_costs_dict[g] = segments

    export_dict_to_temp_json(generator_costs_dict, "linearised_conventional_costs")
    

def get_linearised_conventional_generator_costs():
    with open(os.path.join(os.getcwd(), "bin", "temp", "linearised_conventional_costs.json")) as f:
        data = json.load(f)

    clean_data = {}
    
    for k,v in data.items():
        clean_data[k] = {}
        for kk, vv in v.items():
            if kk != "base_cost_per_hour":
                clean_data[k][int(kk)] = vv
            else:
                clean_data[k][kk] = vv    
    
    return clean_data

def get_result_json(auction_type):
    with open(os.path.join(os.getcwd(), "bin", "temp", f"{auction_type}_auction_results.json")) as f:
        data = json.load(f)

    return data

def set_renewable_generation_data(g_type, max_power):
    df = pd.read_csv(os.path.join(os.getcwd(), "bin", f"{g_type}_generation_main.csv"))
    df = df.set_index("Generator")
    for h in df.columns:
        df[h] = df[h] * max_power
    
    df.to_csv(os.path.join(os.getcwd(), "bin", f"{g_type}_generation.csv"))

def get_individual_operation_costs(auction_type):

    results_data = get_result_json(auction_type)
    gen_lin_costs = get_linearised_conventional_generator_costs()
    gen_dict = get_generator_data()


    gen_final_costs = {}
    gen_power = {}
    for g,v in results_data["gen_units"].items():
        if v["g_type"] == "conventional":
            gen_final_costs[g] = 0
            gen_power[g] = 0
            for t in v.keys():
                if t == "g_type":
                    continue
                
                power = v[t]["generation"]
                gen_power[g] += power
                if power > 0:
                    for segment in gen_lin_costs[g].keys():
                        if segment == "base_cost_per_hour":
                            gen_final_costs[g] += gen_lin_costs[g][segment] * 0.5
                            continue
                        # below segment
                        elif power < gen_lin_costs[g][segment]["lower_limit_mw"]:
                            continue
                        # above segment
                        elif power > gen_lin_costs[g][segment]["upper_limit_mw"]:
                            gen_final_costs[g] += gen_lin_costs[g][segment]["cost_per_mwh"] * (gen_lin_costs[g][segment]["upper_limit_mw"] - gen_lin_costs[g][segment]["lower_limit_mw"]) * 0.5
                        # in segment
                        else:
                            gen_final_costs[g] += gen_lin_costs[g][segment]["cost_per_mwh"] * (power - gen_lin_costs[g][segment]["lower_limit_mw"]) * 0.5
                            

                current = v[t]["is_dispatched"]
                    
                if t == "1":
                    prior = v["48"]["is_dispatched"]
                else:
                    prior = v[str(int(t)-1)]["is_dispatched"]
                
                if current == 1 and prior == 0:
                    gen_final_costs[g] += gen_dict[g]["startup_cost"]
                elif current == 0 and prior == 1:
                    gen_final_costs[g] += gen_dict[g]["shutdown_cost"]
                    

    return gen_final_costs

def get_total_system_cost(auction_type):
    results_data = get_result_json(auction_type)
    results_prices = get_auction_result_prices(auction_type)
    operationsl_costs = get_individual_operation_costs(auction_type)

    power_revenue = {}
    fr_revenue = {}
    sr_revenue = {}
    # startups = 0
    # shutdowns = 0
    for g,v in results_data["gen_units"].items():
        if v["g_type"] in ["solar", "wind", "conventional"]:
            power_key = "generation"
        if v["g_type"] == "storage":
            power_key = "discharge"
        
        power_revenue[g] = 0
        for t in v.keys():
            if t == "g_type":
                continue
            
            power = v[t][power_key]
            price = results_prices.loc[int(t), "Power"]
            power_revenue[g] += power * price            

            # For battery charge
            if v["g_type"] == "storage":
                power_revenue[g] -= v[t]["charge"] * price


    for g,v in results_data["gen_units"].items():
        if v["g_type"] in ["solar", "wind"]:
            continue
        
        fr_revenue[g] = 0
        for t in v.keys():
            if t == "g_type":
                continue
            
            fr_cap = v[t]["fast_reserve"]
            price = results_prices.loc[int(t), "FR"]
            fr_revenue[g] += fr_cap * price

        if v["g_type"] == "conventional":
            sr_revenue[g] = 0
            for t in v.keys():
                if t == "g_type":
                    continue
                
                sr_cap = v[t]["slow_reserve"]
                price = results_prices.loc[int(t), "SR"]
                sr_revenue[g] += sr_cap * price

    no_loss_guarantee = 0
    for g in operationsl_costs.keys():
        total_individual_revenue = power_revenue[g] + fr_revenue[g] + sr_revenue[g]
        profit = total_individual_revenue - operationsl_costs[g]
        no_loss_guarantee += -profit if profit < 0 else 0


    # print("POWER REVENUE")
    # for g in power_revenue.keys():
    #     print(f"{g}-po: {power_revenue[g]:,.2f}")
    
    # print("\nFR REVENUE")
    # for g in fr_revenue.keys():
    #     print(f"{g}-fr: {fr_revenue[g]:,.2f}")
    
    # print("\nSR REVENUE")
    # for g in sr_revenue.keys():
    #     print(f"{g}-sr: {sr_revenue[g]:,.2f}")
    

    total_revenue = sum([power_revenue[g] for g in power_revenue.keys()])
    total_revenue += sum([fr_revenue[g] for g in fr_revenue.keys()])
    total_revenue += sum([sr_revenue[g] for g in sr_revenue.keys()])
    total_revenue += no_loss_guarantee

    print(f"\nTOTAL REVENUE: {total_revenue:,.2f} - {auction_type} auction")
    
    return total_revenue


if __name__ == "__main__":
    get_total_system_cost("cooptimised")
    get_total_system_cost("simultaneous")
    get_total_system_cost("sequential")