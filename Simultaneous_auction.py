import pyomo.environ as pyo
from pyomo.opt import TerminationCondition
import pandas as pd
import random
import os
import utils
import matplotlib.pyplot as plt
import math


def main():

    solver = pyo.SolverFactory("gurobi")
    solver2 = pyo.SolverFactory("gurobi") # glpk

    m = pyo.ConcreteModel()
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    # Trading intervals, 48 30 min intervals
    m.T = pyo.RangeSet(48)
    interval_length = 0.5 # half an hour

    # 30 minute Demand data
    demand = utils.get_demand_data()

    # Generators data--------------------------------------------------------------
    generators_dict = utils.get_generator_data()
    minimum_noload_spinning_capacity_mw = 100 # As MW

    
    # Each generator has a forecasted reserve price, they use it to calculate their offered oppotunity cost
    # Forecasted price is the result from the cooptimised auction plus a random variance for each provider
    forecast_prices_df = utils.get_forecast_prices()
    
    forecast_fast_reserve_price_list = [e[0] for e in forecast_prices_df[["FR"]].values]
    forecast_slow_reserve_price_list = [e[0] for e in forecast_prices_df[["SR"]].values]

    forecast_error = 0.2 # +-20% percent maximum error vs forecast 
    bound_fast = abs(sum(forecast_fast_reserve_price_list) / len(forecast_fast_reserve_price_list)) * forecast_error
    bound_slow = abs(sum(forecast_slow_reserve_price_list) / len(forecast_slow_reserve_price_list)) * forecast_error

    rand_seed = 42
    random.seed(rand_seed)

    # Forecasted reserve prices
    for g in generators_dict.keys():
        rand_seed +=1 
        random.seed(rand_seed)

        # Is the expected reserve price above each generator's reserve cost? If yes, increase energy price to reflect future opportunity, if not, offer only electricity at the production cost
        max_production_mw = generators_dict[g]["max_power_mw"]
        power_production_cost = generators_dict[g]["power_cost_per_mwh"]
        generators_dict[g]["intervals"] = {}

        for t in m.T:
            forecast_fast_reserve_price = forecast_fast_reserve_price_list[t-1] + (random.randint(-math.floor(bound_fast*10), math.ceil(bound_fast*10))/10)
            forecast_slow_reserve_price = forecast_slow_reserve_price_list[t-1] + (random.randint(-math.floor(bound_slow*10), math.ceil(bound_slow*10))/10)
            production_blocks = [max_production_mw,0,0]
            production_blocks_offered_price = [power_production_cost, power_production_cost, power_production_cost]

            # Decide which reserve is more profitable
            fast_reserve_profit = forecast_fast_reserve_price - generators_dict[g]["fast_reserve_cost_per_mw"]
            slow_reserve_profit = forecast_slow_reserve_price - generators_dict[g]["slow_reserve_cost_per_mw"]

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

    # print(generators_dict["G01"])
    # print(generators_dict["G02"])
    # print(generators_dict["G03"])

    m.GENERATORS = pyo.Set(initialize=generators_dict.keys())

    m.generation = pyo.Var(m.GENERATORS, m.T, domain=pyo.NonNegativeReals)
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
    total_cost_generators = sum(generators_dict[g]["intervals"][t]["production_blocks_offered_price"][0] * m.generation_in_blocks[g,t,0] * interval_length for g in m.GENERATORS for t in m.T)
    total_cost_generators += sum(generators_dict[g]["intervals"][t]["production_blocks_offered_price"][1] * m.generation_in_blocks[g,t,1] * interval_length for g in m.GENERATORS for t in m.T)
    total_cost_generators += sum(generators_dict[g]["intervals"][t]["production_blocks_offered_price"][2] * m.generation_in_blocks[g,t,2] * interval_length for g in m.GENERATORS for t in m.T)

    total_cost_wind_generators = sum(wind_generation_var_cost * m.wind_generation[w,t] * interval_length for w in m.WIND_GENERATORS for t in m.T)
    
    total_cost_solar_generators = sum(solar_generation_var_cost * m.solar_generation[spv,t] * interval_length for spv in m.SOLAR_GENERATORS for t in m.T)
    
    total_cost_storage_charge = - sum(storage_dict[s]["charge_price"] * m.storage_charge_power[s,t] * interval_length for s in m.STORAGE for t in m.T)
    total_cost_storage_discharge = sum(storage_dict[s]["discharge_price"] * m.storage_discharge_power[s,t] * interval_length for s in m.STORAGE for t in m.T)


    m.obj = pyo.Objective(
        expr = 
            # Energy cost
            total_cost_generators 
            + total_cost_wind_generators 
            + total_cost_solar_generators 
            + total_cost_storage_charge 
            + total_cost_storage_discharge
            
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

            spinning_capacity += (generators_dict[g]["max_power_mw"] * m.generation_is_dispatched[g,t]) - (m.generation[g, t] * m.generation_is_dispatched[g,t])
        
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
            # return m.generation[g,t] <= generators_dict[g]["max_power_mw"]
            return pyo.Constraint.Skip
        
        return m.generation[g,t] - m.generation[g,t-1] <= generators_dict[g]["max_power_mw"] * (1/generators_dict[g]["total_power_hours"])

    # Ramp down constraint
    @m.Constraint(m.GENERATORS, m.T)
    def gen_ramp_down(m, g, t):
        if t == 1:
            # return m.generation[g,t] <= generators_dict[g]["max_power_mw"] * (1 + (1/generators_dict[g]["total_power_hours"]))
            return pyo.Constraint.Skip
        
        return m.generation[g,t-1] - m.generation[g,t] <= generators_dict[g]["max_power_mw"] * (1/generators_dict[g]["total_power_hours"])


    # Wind generation constraints-----------------
    #
    # Maximum power output
    @m.Constraint(m.WIND_GENERATORS, m.T)
    def wind_gen_max_power(m, w, t):
        # if the forecasted energy is zero, force the generation to be zero
        if wind_generation.loc[t][w] == 0:
            m.wind_generation[w,t] == 0
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
        return m.storage_energy[s,48] == soc_end_of_day * storage_dict[s]["max_capacity_mwh"]

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




    # Marginal price
    marginal_price_power = [m.dual[m.power_balance[t]]/interval_length for t in m.T]
    marginal_price_power_df = pd.DataFrame(zip(m.T, marginal_price_power), columns=["Interval", "Power"])
    marginal_price_power_df.set_index("Interval", inplace=True)

    # marginal_price_fast_reserve = [m.dual[m.fast_reserve_balance[t]]/interval_length for t in m.T]
    # marginal_price_fast_reserve_df = pd.DataFrame(zip(m.T, marginal_price_fast_reserve), columns=["Interval", "Marginal Price FR"])
    # marginal_price_fast_reserve_df.set_index("Interval", inplace=True)

    # marginal_price_power_df["FR"] = marginal_price_fast_reserve_df["Marginal Price FR"]

    # marginal_price_slow_reserve = [m.dual[m.slow_reserve_balance[t]]/interval_length for t in m.T]
    # marginal_price_slow_reserve_df = pd.DataFrame(zip(m.T, marginal_price_slow_reserve), columns=["Interval", "Marginal Price SR"])
    # marginal_price_slow_reserve_df.set_index("Interval", inplace=True)

    # marginal_price_power_df["SR"] = marginal_price_slow_reserve_df["Marginal Price SR"]


    # for g in generators_dict.keys():
    #     marginal_price_power_df[g] = generators_dict[g]["price"]

    # for s in storage_dict.keys():
    #     marginal_price_power_df[f"{s} ch"] = storage_dict[s]["charge_price"]
    #     marginal_price_power_df[f"{s} dis"] = storage_dict[s]["discharge_price"]


    # print(marginal_price_power_df)


    # lineplot of generation versus time

    # print("\n\n")
    storage_energy = pd.Series(m.storage_energy.get_values()).unstack(0)
    storage_charge_power = pd.Series(m.storage_charge_power.get_values()).unstack(0)
    storage_discharge_power = pd.Series(m.storage_discharge_power.get_values()).unstack(0)

    gen = pd.Series(m.generation.get_values()).unstack(0)
    wind_gen = pd.Series(m.wind_generation.get_values()).unstack(0)
    solar_gen = pd.Series(m.solar_generation.get_values()).unstack(0)

    # df = pd.concat([gen, wind_gen, solar_gen], axis=1)
    df = pd.concat([gen, wind_gen, solar_gen, -storage_charge_power, storage_discharge_power], axis=1)
    # fig, axs = plt.subplots(2, 2)
    # df.plot(kind='line', ax=axs[1,0])
    # df.plot(kind='area', stacked=True, ax=axs[1,1])

    # storage_energy.plot(ax=axs[0,1], label='Storage Energy')
    # storage_charge_power.plot(ax=axs[0,1], label='Charge Power')
    # storage_discharge_power.plot(ax=axs[0,1], label='Discharge Power')


    # marginal_price_power_df.plot(kind='line', ax=axs[0,0])

    # plt.show()


    # # blocks and prices from G01
    # intervals = {}
    # for t in m.T:
    #     blocks = generators_dict["G01"]["intervals"][t]["production_blocks"]
    #     prices = generators_dict["G01"]["intervals"][t]["production_blocks_offered_price"]
    #     intervals[t] = prices

    # # plot intervals
    # print(intervals)
    # plt.plot(data=intervals)
    # plt.show()





    print("\n\n------------ AUCTIONING RESERVES ------------\n")

    # Reserves can only be offered by generators that are dispatched and storage (Fast reserve only)


    m2 = pyo.ConcreteModel()
    m2.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    # Trading intervals, 48 30 min intervals
    m2.T = pyo.RangeSet(48)


    # Reserve demand calculation, as a function of demand and renewable generation
    wind_generation = [0 for _ in m.T]
    solar_generation = [0 for _ in m.T]
    renewable_generation = [0 for _ in m.T]

    # Get dispatched renewable generation for variable reserve demand calculation
    for t in m.T:
        for w in m.WIND_GENERATORS:
            wind_generation[t-1] += m.wind_generation[w, t].value * m.wind_generation_percentage[w, t].value

        for pv in m.SOLAR_GENERATORS:
            if m.solar_generation[pv, t].value == 0:
                continue
            solar_generation[t-1] += m.solar_generation[pv, t].value * m.solar_generation_percentage[pv, t].value

        renewable_generation[t-1] = wind_generation[t-1] + solar_generation[t-1]


    # print(renewable_generation)
    fast_reserve_demand = [100 + d*0 + r*0 for d,r in zip(demand, renewable_generation)]
    slow_reserve_demand = [100 + d*0 + r*0 for d,r in zip(demand, renewable_generation)]


    # Generators data--------------------------------------------------------------

    # Get available capacity from generators to offer reserves
    generation_power_dispatch = m.generation.get_values()
    generation_is_dispatched = m.generation_is_dispatched.get_values()

    for g in generators_dict.keys():
        generators_dict[g]["available capacity for reserves"] = {}
        for t in m.T:
            # If a generator is not dispatching power, it cannot offer reserves
            if generation_is_dispatched[g,t] == 0:
                generators_dict[g]["available capacity for reserves"][t] = 0 
            else:
                generators_dict[g]["available capacity for reserves"][t] = generators_dict[g]["max_power_mw"] - generation_power_dispatch[g,t]


    # capacity_for_reserves = []
    # for t in m.T:
    #     capacity_for_reserves.append(sum([generators_dict[g]["available capacity for reserves"][t] for g in generators_dict.keys()]))

    # for t in m.T:
    #     print(capacity_for_reserves[t-1] > fast_reserve_demand[t-1] + slow_reserve_demand[t-1], capacity_for_reserves[t-1], fast_reserve_demand[t-1] + slow_reserve_demand[t-1])
    # raise
    
    m2.GENERATORS = pyo.Set(initialize=generators_dict.keys())
    m2.generation_fast_reserve = pyo.Var(m2.GENERATORS, m2.T, domain=pyo.NonNegativeReals)
    m2.generation_slow_reserve = pyo.Var(m2.GENERATORS, m2.T, domain=pyo.NonNegativeReals)



    # Storage --------------------------------------------------------------

    m2.STORAGE = pyo.Set(initialize=storage_dict.keys())
    m2.storage_fast_reserve_capacity = pyo.Var(m2.STORAGE, m2.T, domain=pyo.NonNegativeReals)


    # Objective function--------------------------------------------------------------
    fast_reserve_cost_generators = sum(generators_dict[k]["fast_reserve_cost_per_mw"] * m2.generation_fast_reserve[k,t] * interval_length for k in m2.GENERATORS for t in m2.T)
    fast_reserve_cost_storage = sum(storage_dict[k]["fast_reserve_price"] * m2.storage_fast_reserve_capacity[k,t] * interval_length for k in m2.STORAGE for t in m2.T)

    slow_reserve_cost_generators = sum(generators_dict[k]["slow_reserve_cost_per_mw"] * m2.generation_slow_reserve[k,t] * interval_length for k in m2.GENERATORS for t in m2.T)


    m2.obj = pyo.Objective(
        expr = 
            # Energy cost
            + fast_reserve_cost_generators
            + fast_reserve_cost_storage

            + slow_reserve_cost_generators
            
            , sense=pyo.minimize)
    # m.pprint()

    # # Constraints--------------------------------------------------------------

    # Fast reserve fulfilment balance
    @m2.Constraint(m2.T)
    def fast_reserve_balance(m2, t):
        reserved_mw = 0
        reserved_mw += sum(m2.generation_fast_reserve[g_name, t] for g_name in m2.GENERATORS)
        reserved_mw += sum(m2.storage_fast_reserve_capacity[s_name, t] for s_name in m2.STORAGE)

        return reserved_mw >= fast_reserve_demand[t - 1]
    # m2.fast_reserve_balance.pprint()


    # Slow reserve fulfilment balance
    @m2.Constraint(m2.T)
    def slow_reserve_balance(m2, t):
        reserved_mw = 0
        reserved_mw += sum(m2.generation_slow_reserve[g_name, t] for g_name in m2.GENERATORS)
        
        return reserved_mw >= slow_reserve_demand[t - 1]
    # m2.power_balance.pprint()

    # Generation constraints-----------------
    #

    # Maximum reserve restriction
    @m2.Constraint(m2.GENERATORS, m2.T)
    def gen_max_fast_reserve(m2, g, t):
        return m2.generation_fast_reserve[g,t] <= generators_dict[g]["max_fast_reserve_mw"]

    @m2.Constraint(m2.GENERATORS, m2.T)
    def gen_max_slow_reserve(m2, g, t):
        return m2.generation_slow_reserve[g,t] + m2.generation_fast_reserve[g,t] <= generators_dict[g]["max_slow_reserve_mw"]



    # Maximum capacity restriction
    @m2.Constraint(m2.GENERATORS, m2.T)
    def gen_max_capacity(m2, g, t):
        return generation_power_dispatch[g,t] + m2.generation_fast_reserve[g,t] + m2.generation_slow_reserve[g,t] <= generators_dict[g]["max_power_mw"] * m.generation_is_dispatched[g,t]


    # Storage constraints-----------------
    #
    # Storage reserve constraints-----------------
    # Maximum upward power capacity
    @m2.Constraint(m2.STORAGE, m2.T)
    def storage_max_reserve_available(m2, s, t):
        return m.storage_discharge_power[s,t].value - m.storage_charge_power[s,t].value + m2.storage_fast_reserve_capacity[s,t] <= storage_dict[s]["max_power_mw"]


    # Maximum next-interval energy storage
    @m2.Constraint(m2.STORAGE, m2.T)
    def storage_min_energy_available(m2, s, t):
        if t == 1:
            return pyo.Constraint.Skip 
        eff = storage_dict[s]["efficiency"]
        
        return storage_dict[s]["min_capacity_mwh"] <= m.storage_energy[s,t-1].value + ((eff * m.storage_charge_power[s,t].value) - ((m.storage_discharge_power[s,t].value + m2.storage_fast_reserve_capacity[s,t])/eff))*interval_length




    # m2.pprint()
    # raise

    results = solver2.solve(m2, tee=True)

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
        
    # m2.generation_fast_reserve.pprint()
    # m2.generation_slow_reserve.pprint()


    # Marginal price
    marginal_price_power = [m.dual[m.power_balance[t]]/interval_length for t in m.T]
    marginal_price_power_df = pd.DataFrame(zip(m.T, marginal_price_power), columns=["Interval", "Power"])
    marginal_price_power_df.set_index("Interval", inplace=True)

    marginal_price_fast_reserve = [m2.dual[m2.fast_reserve_balance[t]]/interval_length for t in m2.T]
    marginal_price_fast_reserve_df = pd.DataFrame(zip(m2.T, marginal_price_fast_reserve), columns=["Interval", "Marginal Price FR"])
    marginal_price_fast_reserve_df.set_index("Interval", inplace=True)

    marginal_price_power_df["FR"] = marginal_price_fast_reserve_df["Marginal Price FR"]

    marginal_price_slow_reserve = [m2.dual[m2.slow_reserve_balance[t]]/interval_length for t in m2.T]
    marginal_price_slow_reserve_df = pd.DataFrame(zip(m2.T, marginal_price_slow_reserve), columns=["Interval", "Marginal Price SR"])
    marginal_price_slow_reserve_df.set_index("Interval", inplace=True)

    marginal_price_power_df["SR"] = marginal_price_slow_reserve_df["Marginal Price SR"]


    # print(marginal_price_power_df)


    # lineplot of generation versus time

    # print("\n\n")
    storage_energy = pd.Series(m.storage_energy.get_values()).unstack(0)
    storage_charge_power = pd.Series(m.storage_charge_power.get_values()).unstack(0)
    storage_discharge_power = pd.Series(m.storage_discharge_power.get_values()).unstack(0)

    gen = pd.Series(m.generation.get_values()).unstack(0)
    wind_gen = pd.Series(m.wind_generation.get_values()).unstack(0)
    solar_gen = pd.Series(m.solar_generation.get_values()).unstack(0)

    # df = pd.concat([gen, wind_gen, solar_gen], axis=1)
    df = pd.concat([gen, wind_gen, solar_gen, -storage_charge_power, storage_discharge_power], axis=1)
    fig, axs = plt.subplots(2, 3)
    df.plot(kind='line', ax=axs[1,0])
    df.plot(kind='area', stacked=True, ax=axs[1,1])

    storage_energy.plot(ax=axs[1,2], label='Storage Energy')
    storage_charge_power.plot(ax=axs[1,2], label='Charge Power')
    storage_discharge_power.plot(ax=axs[1,2], label='Discharge Power')


    marginal_price_power_df.plot(kind='line', ax=axs[0,0])

    storage_fast_reserve = pd.Series(m2.storage_fast_reserve_capacity.get_values()).unstack(0)
    gen_fast_reserve = pd.Series(m2.generation_fast_reserve.get_values()).unstack(0)

    df = pd.concat([gen_fast_reserve, storage_fast_reserve], axis=1)
    df.plot(kind='area', stacked=True, ax=axs[0,1])

    # print("\n\n")
    gen_slow_reserve = pd.Series(m2.generation_slow_reserve.get_values()).unstack(0)

    df = pd.concat([gen_slow_reserve], axis=1)
    df.plot(kind='area', stacked=True, ax=axs[0,2])



    if __name__ == "__main__":
        plt.show()

    
if __name__ == "__main__":
    main()