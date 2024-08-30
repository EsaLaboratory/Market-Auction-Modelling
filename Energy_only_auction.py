import pyomo.environ as pyo
from pyomo.opt import TerminationCondition
import pandas as pd
import random
import utils
import matplotlib.pyplot as plt
import math


interval_length = 0.5 # half an hour
interval_num = 48 # 48 periods of 30 minutes

def main(params={}):

    print("---------Runing Energy-only Auction-----------")

    solver = pyo.SolverFactory("gurobi")

    m = pyo.ConcreteModel()
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    # Trading intervals, 48 30 min intervals
    m.T = pyo.RangeSet(interval_num)

    # 30 minute Demand data
    demand = utils.get_demand_data()

    # Get reserves demand to set the minimum spinning reserve needed
    cooptimised_auction_results_dict = utils.import_dict_from_temp_json(file_name="cooptimised_auction_results")
    fast_reserve_demand = [cooptimised_auction_results_dict["main_results"]["demand"][t]["fast"] for t in m.T]
    slow_reserve_demand = [cooptimised_auction_results_dict["main_results"]["demand"][t]["slow"] for t in m.T]

    total_reserve_demand = [fr+sr for fr,sr in zip(fast_reserve_demand, slow_reserve_demand)]

    minimum_noload_spinning_capacity_mw = max(total_reserve_demand)*1 # As MW
    print(minimum_noload_spinning_capacity_mw)
    # Generators data--------------------------------------------------------------
    generators_dict = utils.get_generator_data()
    generators_cost_dict = utils.get_linearised_conventional_generator_costs()

    reserve_price_increase = 1 + params["reserve_price_inc"] if "reserve_price_inc" in params.keys() else 1
    
    # Each generator has a forecasted reserve price, they use it to calculate their offered oppotunity cost
    # Forecasted price is the result from the cooptimised auction plus a random variance for each provider
    forecast_prices_df = utils.get_auction_result_prices(auction_type="cooptimised")
    
    forecast_fast_reserve_price_list = [e[0] for e in forecast_prices_df[["FR"]].values]
    forecast_slow_reserve_price_list = [e[0] for e in forecast_prices_df[["SR"]].values]

    forecast_error = 0.2 # +-20% percent maximum error vs forecast 
    bound_fast = abs(sum(forecast_fast_reserve_price_list) / len(forecast_fast_reserve_price_list)) * forecast_error
    bound_slow = abs(sum(forecast_slow_reserve_price_list) / len(forecast_slow_reserve_price_list)) * forecast_error

    rand_seed = 42

    # Forecasted reserve prices
    for g in generators_dict.keys():
        rand_seed +=1 
        random.seed(rand_seed)

        # Is the expected reserve price above each generator's reserve cost? If yes, increase energy price to reflect future opportunity, if not, offer only electricity at the production cost
        max_production_mw = generators_dict[g]["max_power_mw"]
        generators_dict[g]["intervals"] = {}
        fast_reserve_cost = generators_cost_dict[g][10]["cost_per_mwh"] * 0.02 * reserve_price_increase
        slow_reserve_cost = generators_cost_dict[g][10]["cost_per_mwh"] * 0.018 * reserve_price_increase 

        for t in m.T:
            forecast_fast_reserve_price = forecast_fast_reserve_price_list[t-1] + (random.randint(-math.floor(bound_fast*10), math.ceil(bound_fast*10))/10)
            forecast_slow_reserve_price = forecast_slow_reserve_price_list[t-1] + (random.randint(-math.floor(bound_slow*10), math.ceil(bound_slow*10))/10)
            production_blocks = [max_production_mw,0,0]
            production_blocks_offered_price = [0,0,0]

            # Decide which reserve is more profitable
            fast_reserve_profit = forecast_fast_reserve_price - fast_reserve_cost
            slow_reserve_profit = forecast_slow_reserve_price - slow_reserve_cost

            # If fast reserve is more profitable, give preference to it via capacity.
            if fast_reserve_profit >= slow_reserve_profit:
                if slow_reserve_profit > 0:
                    
                    # Reduce energy in block 0 and increase energy in block 1
                    production_blocks[1] = generators_dict[g]["max_slow_reserve_mw"]
                    production_blocks[0] -= production_blocks[1]

                    # Calculate cost of energy in block, should be equal to the opportunity cost of the reserve profit plus energy production cost
                    production_blocks_offered_price[1] += slow_reserve_profit

                if fast_reserve_profit > 0:
                    
                    # Reduce energy in block 0 and 1 and increase energy in block 2.
                    # Reduction should be made first to block 1, then to block 0
                    production_blocks[2] = generators_dict[g]["max_fast_reserve_mw"]

                    if production_blocks[1] >= production_blocks[2]:
                        production_blocks[1] -= production_blocks[2]
                    
                    else:
                        production_blocks[1] = 0
                        production_blocks[0] = max_production_mw - production_blocks[2]
                
                    # Calculate cost of energy in block, should be equal to the opportunity cost of the reserve profit plus energy production cost
                    production_blocks_offered_price[2] += fast_reserve_profit

            # If slow reserve is more profitable, give preference above fast reserve
            else:
                if fast_reserve_profit > 0:
                    
                    # Reduce energy in block 0 and increase energy in block 2
                    production_blocks[2] = generators_dict[g]["max_fast_reserve_mw"]
                    production_blocks[0] -= production_blocks[2]

                    # Calculate cost of energy in block, should be equal to the opportunity cost of the reserve profit plus energy production cost
                    production_blocks_offered_price[2] += fast_reserve_profit

                if slow_reserve_profit > 0:
                    
                    # Reduce energy in block 0 and 2 and increase energy in block 1.
                    # Reduction should be made first to block 2, then to block 0
                    production_blocks[1] = generators_dict[g]["max_slow_reserve_mw"]

                    if production_blocks[2] >= production_blocks[1]:
                        production_blocks[2] -= production_blocks[1]
                    
                    else:
                        production_blocks[2] = 0
                        production_blocks[0] = max_production_mw - production_blocks[1]
                
                    # Calculate cost of energy in block, should be equal to the opportunity cost of the reserve profit plus energy production cost
                    production_blocks_offered_price[1] += slow_reserve_profit

            generators_dict[g]["intervals"][t] = {}
            generators_dict[g]["intervals"][t]["forecast_fast_reserve_price"] = forecast_fast_reserve_price
            generators_dict[g]["intervals"][t]["forecast_slow_reserve_price"] = forecast_slow_reserve_price
            generators_dict[g]["intervals"][t]["production_blocks"] = production_blocks
            generators_dict[g]["intervals"][t]["production_blocks_offered_price"] = production_blocks_offered_price

    # print(generators_dict["U1"])
    # # print(generators_dict["U2"])
    # # print(generators_dict["U3"])
    # raise
    
    num_generation_segments = 10

    m.GENERATORS = pyo.Set(initialize=generators_dict.keys())
    m.gen_segments = pyo.RangeSet(num_generation_segments)

    m.generation = pyo.Var(m.GENERATORS, m.T, domain=pyo.NonNegativeReals)
    m.generation_segments = pyo.Var(m.GENERATORS, m.T, m.gen_segments, domain=pyo.NonNegativeReals)
    m.generation_in_blocks = pyo.Var(m.GENERATORS, m.T, [0,1,2], domain=pyo.NonNegativeReals)
    m.generation_is_dispatched = pyo.Var(m.GENERATORS, m.T, domain=pyo.Binary)

    # Wind generators data--------------------------------------------------------------
    wind_generation_var_cost = 0

    # import wind generation data from csv with periods in columns and generators in rows
    wind_generation = utils.get_wind_generator_data()

    m.WIND_GENERATORS = pyo.Set(initialize=wind_generation.columns)

    m.wind_generation = pyo.Var(m.WIND_GENERATORS, m.T, domain=pyo.NonNegativeReals)
    m.wind_generation_percentage = pyo.Var(m.WIND_GENERATORS, m.T, bounds=(0,1))


    # Solar generators data--------------------------------------------------------------

    solar_generation_var_cost = 0

    # import solar generation data from csv with periods in columns and generators in rows
    solar_generation = utils.get_solar_generator_data()

    m.SOLAR_GENERATORS = pyo.Set(initialize=solar_generation.columns)

    m.solar_generation = pyo.Var(m.SOLAR_GENERATORS, m.T, domain=pyo.NonNegativeReals)
    m.solar_generation_percentage = pyo.Var(m.SOLAR_GENERATORS, m.T, bounds=(0,1))

    # Storage --------------------------------------------------------------
    storage_dict = utils.get_storage_data()
    soc_start_of_day = 0.5
    soc_end_of_day = 0.5

    m.STORAGE = pyo.Set(initialize=storage_dict.keys())

    m.storage_charge_power = pyo.Var(m.STORAGE, m.T, domain=pyo.NonNegativeReals)
    m.storage_discharge_power = pyo.Var(m.STORAGE, m.T, domain=pyo.NonNegativeReals)
    m.storage_energy = pyo.Var(m.STORAGE, m.T, domain=pyo.NonNegativeReals)
    m.storage_is_discharging = pyo.Var(m.STORAGE, m.T, domain=pyo.Binary)


    # Objective function--------------------------------------------------------------
    startup_cost_generators = 0
    shutdown_cost_generators = 0
    energy_cost_generators = 0

    for g in m.GENERATORS:
        for t in m.T:
            if t == 1:
                # Startup cost
                startup_cost_generators += generators_dict[g]["startup_cost"] * (1 - m.generation_is_dispatched[g,interval_num]) * m.generation_is_dispatched[g,t]
                # Shutdown cost
                shutdown_cost_generators += generators_dict[g]["shutdown_cost"] * m.generation_is_dispatched[g,interval_num] * (1 - m.generation_is_dispatched[g,t])
            else:
                # Startup cost
                startup_cost_generators += generators_dict[g]["startup_cost"] * (1 - m.generation_is_dispatched[g,t-1]) * m.generation_is_dispatched[g,t]
                # Shutdown cost
                shutdown_cost_generators += generators_dict[g]["shutdown_cost"] * m.generation_is_dispatched[g,t-1] * (1 - m.generation_is_dispatched[g,t])
 
            energy_cost_generators += sum(generators_cost_dict[g][gs]["cost_per_mwh"] * m.generation_segments[g,t,gs] for gs in m.gen_segments) * interval_length
            energy_cost_generators += generators_cost_dict[g]["base_cost_per_hour"] * m.generation_is_dispatched[g,t] * interval_length

    energy_cost_generators += sum(generators_dict[g]["intervals"][t]["production_blocks_offered_price"][0] * m.generation_in_blocks[g,t,0] * interval_length for g in m.GENERATORS for t in m.T)
    energy_cost_generators += sum(generators_dict[g]["intervals"][t]["production_blocks_offered_price"][1] * m.generation_in_blocks[g,t,1] * interval_length for g in m.GENERATORS for t in m.T)
    energy_cost_generators += sum(generators_dict[g]["intervals"][t]["production_blocks_offered_price"][2] * m.generation_in_blocks[g,t,2] * interval_length for g in m.GENERATORS for t in m.T)

    energy_cost_wind_generators = sum(wind_generation_var_cost * m.wind_generation[w,t] * interval_length for w in m.WIND_GENERATORS for t in m.T)
    
    energy_cost_solar_generators = sum(solar_generation_var_cost * m.solar_generation[spv,t] * interval_length for spv in m.SOLAR_GENERATORS for t in m.T)
    
    energy_cost_storage_charge = - sum(storage_dict[s]["charge_price"] * m.storage_charge_power[s,t] * interval_length for s in m.STORAGE for t in m.T)
    energy_cost_storage_discharge = sum(storage_dict[s]["discharge_price"] * m.storage_discharge_power[s,t] * interval_length for s in m.STORAGE for t in m.T)


    m.obj = pyo.Objective(
        expr = 
            # Startup and shutdown costs
            startup_cost_generators
            + shutdown_cost_generators

            # Energy cost
            + energy_cost_generators 
            + energy_cost_wind_generators 
            + energy_cost_solar_generators 
            + energy_cost_storage_charge 
            + energy_cost_storage_discharge
            
            , sense=pyo.minimize)
    # m.pprint()

    # Constraints--------------------------------------------------------------

    # Power balance
    @m.Constraint(m.T)
    def power_balance(m, t):
        generation = 0
        generation += sum(m.generation[g_name, t] for g_name in m.GENERATORS)
        generation += sum(m.wind_generation[wg_name, t] for wg_name in m.WIND_GENERATORS)
        generation += sum(m.solar_generation[sg_name, t] for sg_name in m.SOLAR_GENERATORS)
        generation += sum(m.storage_discharge_power[s_name, t] for s_name in m.STORAGE)
        generation -= sum(m.storage_charge_power[s_name, t] for s_name in m.STORAGE)

        return generation == demand[t - 1]
    # m.power_balance.pprint()

    @m.Constraint(m.T)
    def minimum_noload_spinning_capacity_constraint(m, t):

        spinning_capacity = 0
        for g in generators_dict.keys():

            spinning_capacity += (generators_dict[g]["max_power_mw"] * m.generation_is_dispatched[g,t]) - m.generation[g, t]
        
        return spinning_capacity >= minimum_noload_spinning_capacity_mw
    # m.minimum_noload_spinning_capacity_mw.pprint()

    # raise

    # Generation constraints-----------------
    #
    # Maximum power output
    @m.Constraint(m.GENERATORS, m.T)
    def gen_max_power(m, g, t):
        return m.generation[g,t] <= generators_dict[g]["max_power_mw"] * m.generation_is_dispatched[g,t]

    # Minimum power output
    @m.Constraint(m.GENERATORS, m.T)
    def gen_min_power(m, g, t):
        return m.generation[g,t] >= generators_dict[g]["min_power_mw"] * m.generation_is_dispatched[g,t]

    # Generation total generation equal to sum of segments constraint
    @m.Constraint(m.GENERATORS, m.T)
    def gen_total_segments_power(m, g, t):        
        return m.generation[g,t] == generators_dict[g]["min_power_mw"] * m.generation_is_dispatched[g,t] + sum(m.generation_segments[g,t,gs] for gs in m.gen_segments)

    # Generation segments constrained to their limits
    @m.Constraint(m.GENERATORS, m.T, m.gen_segments)
    def gen_each_segment_power(m, g, t, gs):
        return m.generation_segments[g,t,gs] <= generators_cost_dict[g][gs]["capacity_mw"] * m.generation_is_dispatched[g,t]

    # total dispatched power is equal to the sum of the power in the blocks
    @m.Constraint(m.GENERATORS, m.T)
    def gen_total_dispatched_power(m, g, t):
        return m.generation[g,t] == sum(m.generation_in_blocks[g,t,b] for b in [0,1,2])

    # Individual power in each block should be less than the maximum power in that block
    @m.Constraint(m.GENERATORS, m.T)
    def gen_max_block_0_power(m, g, t):
        return m.generation_in_blocks[g,t,0] <= generators_dict[g]["intervals"][t]["production_blocks"][0]

    @m.Constraint(m.GENERATORS, m.T)
    def gen_max_block_1_power(m, g, t):
        return m.generation_in_blocks[g,t,1] <= generators_dict[g]["intervals"][t]["production_blocks"][1]

    @m.Constraint(m.GENERATORS, m.T)
    def gen_max_block_2_power(m, g, t):
        return m.generation_in_blocks[g,t,2] <= generators_dict[g]["intervals"][t]["production_blocks"][2]


    # Ramp up constraint
    @m.Constraint(m.GENERATORS, m.T)
    def gen_ramp_up(m, g, t):
        if t == 1:
            return pyo.Constraint.Skip 
        
        return m.generation[g,t] - m.generation[g,t-1] <= generators_dict[g]["max_power_mw"] * interval_length * (1/generators_dict[g]["total_power_hours"])

    # Ramp down constraint
    @m.Constraint(m.GENERATORS, m.T)
    def gen_ramp_down(m, g, t):
        if t == 1:
            return pyo.Constraint.Skip 
        
        return m.generation[g,t-1] - m.generation[g,t] <= generators_dict[g]["max_power_mw"] * interval_length * (1/generators_dict[g]["total_power_hours"])

    # INITIAL STATE OF GENERATORS
    @m.Constraint(m.GENERATORS)
    def gen_initial_state(m, g):
        if g in ["U1", "U2"]:
            return m.generation_is_dispatched[g,1] == 1
        else:
            return pyo.Constraint.Skip


    # Wind generation constraints-----------------
    #
    # Maximum power output
    @m.Constraint(m.WIND_GENERATORS, m.T)
    def wind_gen_max_power(m, w, t):
        # if the forecasted energy is zero, force the generation to be zero
        if wind_generation.loc[t][w] == 0:
            return m.wind_generation[w,t] == 0
        else:
            return m.wind_generation[w,t] == wind_generation.loc[t][w] * m.wind_generation_percentage[w,t]

    # Solar generation constraints-----------------
    #
    # Maximum power output
    @m.Constraint(m.SOLAR_GENERATORS, m.T)
    def solar_gen_max_power(m, spv, t):
        # if the forecasted energy is zero, force the generation to be zero
        if solar_generation.loc[t][spv] == 0:
            return m.solar_generation[spv,t] == 0
        else:
            return m.solar_generation[spv,t] == solar_generation.loc[t][spv] * m.solar_generation_percentage[spv,t]


    # Storage constraints-----------------
    #
    # Maximum power charging
    @m.Constraint(m.STORAGE, m.T)
    def storage_max_charge_power(m, s, t):
        return m.storage_charge_power[s,t] <= storage_dict[s]["max_power_mw"] * (1 - m.storage_is_discharging[s,t])

    # Maximum power discharging
    @m.Constraint(m.STORAGE, m.T)
    def storage_max_discharge_power(m, s, t):
        return m.storage_discharge_power[s,t] <= storage_dict[s]["max_power_mw"] * m.storage_is_discharging[s,t]

    # Energy storage at the end of the period 
    @m.Constraint(m.STORAGE, m.T)
    def storage_keep_track_of_energy(m, s, t):
        
        eff = storage_dict[s]["efficiency"]
        if t == 1:
            return m.storage_energy[s,t] == (soc_start_of_day * storage_dict[s]["max_capacity_mwh"]) + ((eff * m.storage_charge_power[s,t]) - (m.storage_discharge_power[s,t]/eff))*interval_length
        
        return m.storage_energy[s,t] == m.storage_energy[s,t-1] + ((eff * m.storage_charge_power[s,t]) - (m.storage_discharge_power[s,t]/eff))*interval_length

    # Energy storage lower than maximum capacity
    @m.Constraint(m.STORAGE, m.T)
    def storage_max_energy_capacity(m, s, t):
        return m.storage_energy[s,t] <= storage_dict[s]["max_capacity_mwh"]

    # Energy storage higher than minimum capacity
    @m.Constraint(m.STORAGE, m.T)
    def storage_min_energy_capacity(m, s, t):
        return m.storage_energy[s,t] >= storage_dict[s]["min_capacity_mwh"]
        
    # Energy STORAGE for end of day
    @m.Constraint(m.STORAGE)
    def storage_keep_track_of_energy_end_of_day(m, s):
        return m.storage_energy[s,interval_num] == soc_end_of_day * storage_dict[s]["max_capacity_mwh"]

    # solve the optimization problem
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

    # Fix integer variables and run again to obtain dual variables (Prices)
    for t in m.T:
        for g in m.GENERATORS:
            m.generation_is_dispatched[g,t].fix(m.generation_is_dispatched[g,t].value)
        for s in m.STORAGE:
            m.storage_is_discharging[s,t].fix(m.storage_is_discharging[s,t].value)
        

    # m.pprint()


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


    # Export generation results to a json file

    model_results_dict = {
        "main_results": {},
        "gen_units": {}
    }
    
    for g in m.GENERATORS:
        model_results_dict["gen_units"][g] = {"g_type": "conventional"}
        for t in m.T:
            model_results_dict["gen_units"][g][t] = {}
            model_results_dict["gen_units"][g][t]["generation"] = m.generation[g,t].value
            model_results_dict["gen_units"][g][t]["is_dispatched"] = m.generation_is_dispatched[g,t].value

            # Get available capacity from generators to offer reserves
            # If a generator is not dispatching power, it cannot offer reserves
            if m.generation_is_dispatched[g,t].value == 0:
                model_results_dict["gen_units"][g][t]["available_for_reserves"] = 0 
            else:
                model_results_dict["gen_units"][g][t]["available_for_reserves"] = generators_dict[g]["max_power_mw"] - m.generation[g,t].value

    for w in m.WIND_GENERATORS:
        model_results_dict["gen_units"][w] = {"g_type": "wind"}
        for t in m.T:
            model_results_dict["gen_units"][w][t] = {}
            model_results_dict["gen_units"][w][t]["generation"] = m.wind_generation[w,t].value
            model_results_dict["gen_units"][w][t]["percentage"] = m.wind_generation_percentage[w,t].value

    for pv in m.SOLAR_GENERATORS:
        model_results_dict["gen_units"][pv] = {"g_type": "solar"}
        for t in m.T:
            model_results_dict["gen_units"][pv][t] = {}
            model_results_dict["gen_units"][pv][t]["generation"] = m.solar_generation[pv,t].value
            model_results_dict["gen_units"][pv][t]["percentage"] = m.solar_generation_percentage[pv,t].value
    
    for s in m.STORAGE:
        model_results_dict["gen_units"][s] = {"g_type": "storage"}
        for t in m.T:
            model_results_dict["gen_units"][s][t] = {}
            model_results_dict["gen_units"][s][t]["charge"] = m.storage_charge_power[s,t].value
            model_results_dict["gen_units"][s][t]["discharge"] = m.storage_discharge_power[s,t].value
            model_results_dict["gen_units"][s][t]["energy"] = m.storage_energy[s,t].value

    # Power prices
    marginal_price_power = [m.dual[m.power_balance[t]]/interval_length for t in m.T]
    model_results_dict["main_results"]["marginal_price"] = {}
    for t in m.T:
        model_results_dict["main_results"]["marginal_price"][t] = {"power": marginal_price_power[t-1]}
    
    # System cost
    model_results_dict["main_results"]["system_cost"] = m.obj()

    
    # Reserve demand calculation, as a function of demand and renewable generation
    renewable_generation = [0 for _ in m.T]

    # Get dispatched renewable generation for variable reserve demand calculation
    for k, v in model_results_dict["gen_units"].items():
        g_type = v["g_type"]
        if g_type == "wind" or g_type == "solar":
            for t in m.T:
                renewable_generation[t-1] += v[t]["generation"]
    
    model_results_dict["main_results"]["demand"] = {}
    for t in m.T:
        model_results_dict["main_results"]["demand"][t] = {"power": demand[t-1], "fast": fast_reserve_demand[t-1], "slow": slow_reserve_demand[t-1]}
                

    # Export results to a json file
    print("Exporting results to json file")
    utils.export_dict_to_temp_json(dict_data=model_results_dict, file_name="energy_only_auction_results")  
    

    print("\n---------Energy-only auction finished---------\n\n")
    print(f"system cost: {m.obj():,.0f}")


    # Uncomment this section to extract scheduling information
    # startup_cost_generators = 0
    # shutdown_cost_generators = 0
    # energy_cost_generators = 0

    # reserve_increase_dict = {}
    # for g in m.GENERATORS:
    #     reserve_increase_dict[g] = 0
    #     for t in m.T:
    #         reserve_increase_dict[g] += generators_dict[g]["intervals"][t]["production_blocks_offered_price"][0] * m.generation_in_blocks[g,t,0].value * interval_length 
    #         reserve_increase_dict[g] += generators_dict[g]["intervals"][t]["production_blocks_offered_price"][1] * m.generation_in_blocks[g,t,1].value * interval_length 
    #         reserve_increase_dict[g] += generators_dict[g]["intervals"][t]["production_blocks_offered_price"][2] * m.generation_in_blocks[g,t,2].value * interval_length 

    # for g in generators_dict.keys():
    #     if g == "U1":
    #         print(generators_dict[g])
    
    # values = [(g,val["capacity_mw"], val["cost_per_mwh"]) for g, v in generators_cost_dict.items() for seg, val in v.items() if seg != "base_cost_per_hour"]

    # # order list based in cost values
    # values = sorted(values, key=lambda x: x[2])

    # for a,b,c in values:
    #     print(f"{a},{b},{c}")
    
    # print("\n\n\n-------------------\n\n\n")
    # # raise

    # # Create the supply curve
    # capacities = [0] + [val[1] for val in values]

    # # add all past elements for each element
    # for i in range(1, len(capacities)):
    #     capacities[i] += capacities[i-1]

    # costs = [val[2] for val in values] + [values[-1][2]]

    # interv = 4
    
    # # gen_increments = {g:0 for g in generators_dict.keys()}    
    # new_values = []
    # for g, v in generators_cost_dict.items():
    #     for seg, val in v.items():
    #         if seg != "base_cost_per_hour":
    #             # gen_increments[g] += val["capacity_mw"]

    #             base_cap = generators_dict[g]["intervals"][interv]["production_blocks"][0]
    #             inc_cost = max(generators_dict[g]["intervals"][interv]["production_blocks_offered_price"][1::])

    #             new_values.append((g, val["capacity_mw"], val["cost_per_mwh"] + inc_cost))


    # new_values = sorted(new_values, key=lambda x: x[2])
    # for a,b,c in new_values:
    #     print(f"{a},{b},{c}")
    


if __name__ == "__main__":
    main({"reserve_price_inc": 0})