import pyomo.environ as pyo
from pyomo.opt import TerminationCondition
import pandas as pd
import utils
import matplotlib.pyplot as plt
import random
import math

interval_length = 0.5 # half an hour
interval_num = 48 # 48 periods of 30 minutes

def main():

    print("---------Runing Sequential Reserves Auction-----------")

    # Reserves can only be offered by generators that are dispatched and storage (Fast reserve only)

    solver = pyo.SolverFactory("gurobi")
    m = pyo.ConcreteModel()
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    # Trading intervals, 48 30 min intervals
    m.T = pyo.RangeSet(interval_num)

    # import json with energy auction and simultaneous reserve auction results and convert to dict
    print("Reading json files")
    energy_auction_results_dict = utils.import_dict_from_temp_json(file_name="energy_only_auction_results")
    
    # Reserves demand data--------------------------------------------------------------
    # print(energy_auction_results_dict["main_results"])
    
    fast_reserve_demand = [energy_auction_results_dict["main_results"]["demand"][t]["fast"] for t in m.T]
    slow_reserve_demand = [energy_auction_results_dict["main_results"]["demand"][t]["slow"] for t in m.T]

    # Generators data--------------------------------------------------------------
    generators_dict = utils.get_generator_data()

    m.GENERATORS = pyo.Set(initialize=generators_dict.keys())
    m.generation_fast_reserve = pyo.Var(m.GENERATORS, m.T, domain=pyo.NonNegativeReals)
    m.generation_fast_res_in_blocks = pyo.Var(m.GENERATORS, m.T, [0,1], domain=pyo.NonNegativeReals)


    # Storage --------------------------------------------------------------
    storage_dict = utils.get_storage_data()
    
    m.STORAGE = pyo.Set(initialize=storage_dict.keys())
    m.storage_fast_reserve_capacity = pyo.Var(m.STORAGE, m.T, domain=pyo.NonNegativeReals)


    # Calculate oportunity cost for each generator and storage
    # Each generator has a forecasted slow reserve price, they use it to calculate their offered oppotunity cost
    # Forecasted price is the result from the cooptimised auction plus a random variance for each provider
    forecast_prices_df = utils.get_auction_result_prices(auction_type="cooptimised") # Check later with simultaneous auction
    
    forecast_slow_reserve_price_list = [e[0] for e in forecast_prices_df[["SR"]].values]

    forecast_error = 0.2 # +-20% percent maximum error vs forecast 
    bound_slow = abs(sum(forecast_slow_reserve_price_list) / len(forecast_slow_reserve_price_list)) * forecast_error

    rand_seed = 42

    # Forecasted reserve prices
    for g in generators_dict.keys():
        rand_seed +=1 
        random.seed(rand_seed)

        # Is the expected slow reserve price above each generator's slow reserve cost? 
        # If yes, increase fast reserve price to reflect future opportunity, 
        # If not, offer only fast reserve at true cost
        generators_dict[g]["intervals"] = {}
        max_slow_reserve = generators_dict[g]["max_slow_reserve_mw"]
        fast_reserve_cost = generators_dict[g]["fast_reserve_cost_per_mw"]
        slow_reserve_cost = generators_dict[g]["slow_reserve_cost_per_mw"]

        for t in m.T:
            cap_available_for_reserves_mw = energy_auction_results_dict["gen_units"][g][t]["available_for_reserves"]
            forecast_slow_reserve_price = forecast_slow_reserve_price_list[t-1] + (random.randint(-math.floor(bound_slow*10), math.ceil(bound_slow*10))/10)
            
            reserve_blocks = [cap_available_for_reserves_mw,0]
            reserve_blocks_offered_price = [fast_reserve_cost, fast_reserve_cost]

            # Potential profit of slow reserve
            slow_reserve_profit = forecast_slow_reserve_price - slow_reserve_cost

            if slow_reserve_profit > 0:
                
                # Reduce energy in block 0 and increase energy in block 1
                reserve_blocks[1] = min(max_slow_reserve, cap_available_for_reserves_mw)
                reserve_blocks[0] -= min(max_slow_reserve, cap_available_for_reserves_mw)

                # Calculate cost of energy in block, should be equal to the opportunity cost of the reserve profit plus energy production cost
                reserve_blocks_offered_price[1] += slow_reserve_profit

            generators_dict[g]["intervals"][t] = {}
            generators_dict[g]["intervals"][t]["available_for_reserves"] = cap_available_for_reserves_mw
            generators_dict[g]["intervals"][t]["forecast_slow_reserve_price"] = forecast_slow_reserve_price
            generators_dict[g]["intervals"][t]["reserve_blocks"] = reserve_blocks
            generators_dict[g]["intervals"][t]["reserve_blocks_offered_price"] = reserve_blocks_offered_price


    # print(generators_dict["G01"],"\n\n")
    # print(generators_dict["G02"],"\n\n")
    # print(generators_dict["G03"],"\n\n")

    # Objective function--------------------------------------------------------------
    fast_reserve_cost_generators = sum(generators_dict[g]["intervals"][t]["reserve_blocks_offered_price"][0] * m.generation_fast_res_in_blocks[g,t,0] * interval_length for g in m.GENERATORS for t in m.T)
    fast_reserve_cost_generators += sum(generators_dict[g]["intervals"][t]["reserve_blocks_offered_price"][1] * m.generation_fast_res_in_blocks[g,t,1] * interval_length for g in m.GENERATORS for t in m.T)
    
    fast_reserve_cost_storage = sum(storage_dict[k]["fast_reserve_price"] * m.storage_fast_reserve_capacity[k,t] * interval_length for k in m.STORAGE for t in m.T)

    m.obj = pyo.Objective(
        expr = 
            # Fast reserve cost
            + fast_reserve_cost_generators
            + fast_reserve_cost_storage

            , sense=pyo.minimize)
    # m.pprint()

    # Constraints--------------------------------------------------------------

    # Fast reserve fulfilment balance
    @m.Constraint(m.T)
    def fast_reserve_balance(m, t):
        reserved_mw = 0
        reserved_mw += sum(m.generation_fast_reserve[g_name, t] for g_name in m.GENERATORS)
        reserved_mw += sum(m.storage_fast_reserve_capacity[s_name, t] for s_name in m.STORAGE)

        return reserved_mw >= fast_reserve_demand[t - 1]
    # m.fast_reserve_balance.pprint()


    # Generation constraints-----------------
    #

    # Maximum reserve restriction
    @m.Constraint(m.GENERATORS, m.T)
    def gen_max_fast_reserve(m, g, t):
        return m.generation_fast_reserve[g,t] <= generators_dict[g]["max_fast_reserve_mw"]

    # total dispatched power is equal to the sum of the power in the blocks
    @m.Constraint(m.GENERATORS, m.T)
    def gen_total_fast_reserve(m, g, t):
        return m.generation_fast_reserve[g,t] == sum(m.generation_fast_res_in_blocks[g,t,b] for b in [0,1])

    # Individual power in each block should be less than the maximum power in that block
    @m.Constraint(m.GENERATORS, m.T)
    def gen_max_block_0_fast_reserve(m, g, t):
        return m.generation_fast_res_in_blocks[g,t,0] <= generators_dict[g]["intervals"][t]["reserve_blocks"][0]

    @m.Constraint(m.GENERATORS, m.T)
    def gen_max_block_1_fast_reserve(m, g, t):
        return m.generation_fast_res_in_blocks[g,t,1] <= generators_dict[g]["intervals"][t]["reserve_blocks"][1]



    # Maximum capacity restriction
    @m.Constraint(m.GENERATORS, m.T)
    def gen_max_capacity(m, g, t):
        return energy_auction_results_dict["gen_units"][g][t]["generation"] + m.generation_fast_reserve[g,t] <= generators_dict[g]["max_power_mw"] * energy_auction_results_dict["gen_units"][g][t]["is_dispatched"]

    # Storage constraints-----------------
    #
    # Storage reserve constraints-----------------
    # Maximum upward power capacity
    @m.Constraint(m.STORAGE, m.T)
    def storage_max_reserve_available(m, s, t):
        return energy_auction_results_dict["gen_units"][s][t]["discharge"] - energy_auction_results_dict["gen_units"][s][t]["charge"] + m.storage_fast_reserve_capacity[s,t] <= storage_dict[s]["max_power_mw"]


    # Maximum next-interval energy storage
    @m.Constraint(m.STORAGE, m.T)
    def storage_min_energy_available(m, s, t):
        if t == 1:
            return pyo.Constraint.Skip 
        eff = storage_dict[s]["efficiency"]
        
        return storage_dict[s]["min_capacity_mwh"] <= energy_auction_results_dict["gen_units"][s][t-1]["energy"] + ((eff * energy_auction_results_dict["gen_units"][s][t]["charge"]) - ((energy_auction_results_dict["gen_units"][s][t]["discharge"] + m.storage_fast_reserve_capacity[s,t])/eff))*interval_length

    # m.pprint()
    # raise

    results = solver.solve(m, tee=True)

    # ALWAYS check solver's termination condition
    if results.solver.termination_condition != TerminationCondition.optimal:
        raise Exception
    else:
        print("-------------------")
        print(results.solver.status)
        print(results.solver.termination_condition)
        print(results.solver.termination_message)
        print(results.solver.time)
        print("-------------------")

    print("\n---------Fast reserve auction finished---------\n\n")
    

    # Calculate remaining available capacity for slow reserve of each generator
    avail = {}
    for t in m.T:
        avail[t] = 0
        for g in m.GENERATORS:
            generators_dict[g]["intervals"][t]["available_for_slow_reserves"] = generators_dict[g]["intervals"][t]["available_for_reserves"] - m.generation_fast_reserve[g,t].value
            avail[t] += generators_dict[g]["intervals"][t]["available_for_slow_reserves"]

    # for t in m.T:
    #     print(avail[t]>= slow_reserve_demand[t-1], avail[t], slow_reserve_demand[t-1])
    
    



    # slow reserve individual auction--------------------------------------------------------------
    solver2 = pyo.SolverFactory("gurobi")
    m2 = pyo.ConcreteModel()
    m2.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    # Trading intervals, 48 30 min intervals
    m2.T = pyo.RangeSet(interval_num)

    # Generators data--------------------------------------------------------------
    m2.GENERATORS = pyo.Set(initialize=generators_dict.keys())
    m2.generation_slow_reserve = pyo.Var(m2.GENERATORS, m2.T, domain=pyo.NonNegativeReals)

    # Objective function--------------------------------------------------------------
    slow_reserve_cost_generators = sum(generators_dict[k]["slow_reserve_cost_per_mw"] * m2.generation_slow_reserve[k,t] * interval_length for k in m2.GENERATORS for t in m2.T)


    m2.obj = pyo.Objective(
        expr = 
            # Slow reserve cost
            + slow_reserve_cost_generators

            , sense=pyo.minimize)
    # m2.pprint()

    # Constraints--------------------------------------------------------------

    # Slow reserve fulfilment balance
    @m2.Constraint(m2.T)
    def slow_reserve_balance(m2, t):
        reserved_mw = 0
        reserved_mw += sum(m2.generation_slow_reserve[g_name, t] for g_name in m2.GENERATORS)
        
        return reserved_mw >= slow_reserve_demand[t - 1]
    # m2.slow_balance.pprint()


    # Generation constraints-----------------
    #

    # Maximum reserve restriction
    @m2.Constraint(m2.GENERATORS, m2.T)
    def gen_max_slow_reserve(m2, g, t):
        return m2.generation_slow_reserve[g,t] <= generators_dict[g]["max_slow_reserve_mw"]


    # Maximum capacity restriction
    @m2.Constraint(m2.GENERATORS, m2.T)
    def gen_max_capacity(m2, g, t):
        return m2.generation_slow_reserve[g,t] <= generators_dict[g]["intervals"][t]["available_for_slow_reserves"]

    # m2.pprint()
    # raise

    results2 = solver2.solve(m2, tee=True)

    # ALWAYS check solver's termination condition
    if results2.solver.termination_condition != TerminationCondition.optimal:
        raise Exception
    else:
        print("-------------------")
        print(results2.solver.status)
        print(results2.solver.termination_condition)
        print(results2.solver.termination_message)
        print(results2.solver.time)
        print("-------------------")



    # Calculate the total cost of the system
    total_cost = energy_auction_results_dict["main_results"]["system_cost"] + m.obj() + m2.obj()
    print(f"Total cost: {total_cost:,.0f}")
    

    # Marginal price extraction
    marginal_price_power = [v["power"] for _, v in energy_auction_results_dict["main_results"]["marginal_price"].items()]
    marginal_price_power_df = pd.DataFrame(zip(m.T, marginal_price_power), columns=["Interval", "Power"])
    marginal_price_power_df.set_index("Interval", inplace=True)
    
    marginal_price_fast_reserve = [m.dual[m.fast_reserve_balance[t]]/interval_length for t in m.T]
    marginal_price_fast_reserve_df = pd.DataFrame(zip(m.T, marginal_price_fast_reserve), columns=["Interval", "Marginal Price FR"])
    marginal_price_fast_reserve_df.set_index("Interval", inplace=True)

    marginal_price_power_df["FR"] = marginal_price_fast_reserve_df["Marginal Price FR"]

    marginal_price_slow_reserve = [m2.dual[m2.slow_reserve_balance[t]]/interval_length for t in m2.T]
    marginal_price_slow_reserve_df = pd.DataFrame(zip(m2.T, marginal_price_slow_reserve), columns=["Interval", "Marginal Price SR"])
    marginal_price_slow_reserve_df.set_index("Interval", inplace=True)

    marginal_price_power_df["SR"] = marginal_price_slow_reserve_df["Marginal Price SR"]

    utils.write_temp_price_file(df=marginal_price_power_df, auction_type="sequential")

    # Assemble final results dict ------------------------------
    final_results_dict = energy_auction_results_dict.copy()

    # Add reserve prices and reserve demand
    for t in m.T:
        final_results_dict["main_results"]["marginal_price"][t]["FR"] = marginal_price_fast_reserve[t-1]
        final_results_dict["main_results"]["marginal_price"][t]["SR"] = marginal_price_slow_reserve[t-1]
        
    # Update total system cost
    final_results_dict["main_results"]["system_cost"] = total_cost
    
    
    # Add reserved capacity for each generator
    for g in m.GENERATORS:
        for t in m.T:
            final_results_dict["gen_units"][g][t]["fast_reserve"] = m.generation_fast_reserve[g,t].value
            final_results_dict["gen_units"][g][t]["slow_reserve"] = m2.generation_slow_reserve[g,t].value

    # Add reserved capacity for each storage
    for s in m.STORAGE:
        for t in m.T:
            final_results_dict["gen_units"][s][t]["fast_reserve"] = m.storage_fast_reserve_capacity[s,t].value

    # Export results to a json file
    print("Exporting results to json file")
    utils.export_dict_to_temp_json(dict_data=final_results_dict, file_name="sequential_auction_results")
    
    # lineplot of generation versus time
    if __name__ == "__main__":
        print("\n\n")
        storage_energy = pd.Series({(k,t):v[t]["energy"] for t in m.T for k, v in energy_auction_results_dict["gen_units"].items() if v["g_type"] == "storage"}).unstack(0)
        storage_charge_power = pd.Series({(k,t):v[t]["charge"] for t in m.T for k, v in energy_auction_results_dict["gen_units"].items() if v["g_type"] == "storage"}).unstack(0)
        storage_discharge_power = pd.Series({(k,t):v[t]["discharge"] for t in m.T for k, v in energy_auction_results_dict["gen_units"].items() if v["g_type"] == "storage"}).unstack(0)


        gen = pd.Series({(k,t):v[t]["generation"] for t in m.T for k, v in energy_auction_results_dict["gen_units"].items() if v["g_type"] == "conventional"}).unstack(0)
        wind_gen = pd.Series({(k,t):v[t]["generation"] for t in m.T for k, v in energy_auction_results_dict["gen_units"].items() if v["g_type"] == "wind"}).unstack(0)
        solar_gen = pd.Series({(k,t):v[t]["generation"] for t in m.T for k, v in energy_auction_results_dict["gen_units"].items() if v["g_type"] == "solar"}).unstack(0)
        

        # df = pd.concat([gen, wind_gen, solar_gen], axis=1)
        df = pd.concat([gen, wind_gen, solar_gen, -storage_charge_power, storage_discharge_power], axis=1)
        fig, axs = plt.subplots(2, 3)
        df.plot(kind='line', ax=axs[1,0])
        df.plot(kind='area', stacked=True, ax=axs[1,1])

        storage_energy.plot(ax=axs[1,2], label='Storage Energy')
        storage_charge_power.plot(ax=axs[1,2], label='Charge Power')
        storage_discharge_power.plot(ax=axs[1,2], label='Discharge Power')


        marginal_price_power_df.plot(kind='line', ax=axs[0,0])

        storage_fast_reserve = pd.Series(m.storage_fast_reserve_capacity.get_values()).unstack(0)
        gen_fast_reserve = pd.Series(m.generation_fast_reserve.get_values()).unstack(0)

        df = pd.concat([gen_fast_reserve, storage_fast_reserve], axis=1)
        df.plot(kind='area', stacked=True, ax=axs[0,1])

        gen_slow_reserve = pd.Series(m2.generation_slow_reserve.get_values()).unstack(0)

        df = pd.concat([gen_slow_reserve], axis=1)
        df.plot(kind='area', stacked=True, ax=axs[0,2])

        plt.show()

    print("\n---------Sequential reserve auction finished---------\n\n")


if __name__ == "__main__":
    main()