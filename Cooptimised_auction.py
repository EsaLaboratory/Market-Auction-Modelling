import pyomo.environ as pyo
from pyomo.opt import SolverFactory, TerminationCondition
import pandas as pd
import utils
import os
import matplotlib.pyplot as plt


def main():
    solver = pyo.SolverFactory("gurobi")

    m = pyo.ConcreteModel()
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    # Trading intervals, 48 30 min intervals
    m.T = pyo.RangeSet(48)
    interval_length = 0.5 # half an hour

    # 30 minute Demand data
    demand = utils.get_demand_data()

    # Reserve demand data
    fast_reserve_demand = [100 + d*0.05 for d in demand]
    slow_reserve_demand = [100 + d*0.05 for d in demand]


    # Generators data--------------------------------------------------------------
    generators_dict = utils.get_generator_data()

    m.GENERATORS = pyo.Set(initialize=generators_dict.keys())

    m.generation = pyo.Var(m.GENERATORS, m.T, domain=pyo.NonNegativeReals)
    m.generation_is_dispatched = pyo.Var(m.GENERATORS, m.T, domain=pyo.Binary)
    m.generation_fast_reserve = pyo.Var(m.GENERATORS, m.T, domain=pyo.NonNegativeReals)
    m.generation_slow_reserve = pyo.Var(m.GENERATORS, m.T, domain=pyo.NonNegativeReals)


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
    m.storage_fast_reserve_capacity = pyo.Var(m.STORAGE, m.T, domain=pyo.NonNegativeReals)
    m.storage_energy = pyo.Var(m.STORAGE, m.T, domain=pyo.NonNegativeReals)
    m.storage_is_discharging = pyo.Var(m.STORAGE, m.T, domain=pyo.Binary)


    # Objective function--------------------------------------------------------------
    total_cost_generators = sum(generators_dict[k]["power_cost_per_mwh"] * m.generation[k,t] * interval_length for k in m.GENERATORS for t in m.T)
    total_cost_wind_generators = sum(wind_generation_var_cost * m.wind_generation[k,t] * interval_length for k in m.WIND_GENERATORS for t in m.T)
    total_cost_solar_generators = sum(solar_generation_var_cost * m.solar_generation[k,t] * interval_length for k in m.SOLAR_GENERATORS for t in m.T)
    total_cost_storage_charge = -sum(storage_dict[k]["charge_price"] * m.storage_charge_power[k,t] * interval_length for k in m.STORAGE for t in m.T)
    total_cost_storage_discharge = sum(storage_dict[k]["discharge_price"] * m.storage_discharge_power[k,t] * interval_length for k in m.STORAGE for t in m.T)

    fast_reserve_cost_generators = sum(generators_dict[k]["fast_reserve_cost_per_mw"] * m.generation_fast_reserve[k,t] * interval_length for k in m.GENERATORS for t in m.T)
    fast_reserve_cost_storage = sum(storage_dict[k]["fast_reserve_price"] * m.storage_fast_reserve_capacity[k,t] * interval_length for k in m.STORAGE for t in m.T)

    slow_reserve_cost_generators = sum(generators_dict[k]["slow_reserve_cost_per_mw"] * m.generation_slow_reserve[k,t] * interval_length for k in m.GENERATORS for t in m.T)


    m.obj = pyo.Objective(
        expr = 
            # Energy cost
            total_cost_generators 
            + total_cost_wind_generators 
            + total_cost_solar_generators 
            + total_cost_storage_charge 
            + total_cost_storage_discharge
            
            # Reserve cost
            + fast_reserve_cost_generators
            + fast_reserve_cost_storage

            + slow_reserve_cost_generators

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

    # Fast reserve fulfilment balance
    @m.Constraint(m.T)
    def fast_reserve_balance(m, t):
        reserved_mw = 0
        reserved_mw += sum(m.generation_fast_reserve[g_name, t] for g_name in m.GENERATORS)
        reserved_mw += sum(m.storage_fast_reserve_capacity[s_name, t] for s_name in m.STORAGE)

        return reserved_mw >= fast_reserve_demand[t - 1]
    # m.power_balance.pprint()

    # Slow reserve fulfilment balance
    @m.Constraint(m.T)
    def slow_reserve_balance(m, t):
        reserved_mw = 0
        reserved_mw += sum(m.generation_slow_reserve[g_name, t] for g_name in m.GENERATORS)
        
        return reserved_mw >= slow_reserve_demand[t - 1]
    # m.power_balance.pprint()


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


    # Ramp up constraint
    @m.Constraint(m.GENERATORS, m.T)
    def gen_ramp_up(m, g, t):
        if t == 1:
            return pyo.Constraint.Skip 
        
        return m.generation[g,t] - m.generation[g,t-1] <= generators_dict[g]["max_power_mw"] * (1/generators_dict[g]["total_power_hours"])

    # Ramp down constraint
    @m.Constraint(m.GENERATORS, m.T)
    def gen_ramp_down(m, g, t):
        if t == 1:
            return pyo.Constraint.Skip 
        
        return m.generation[g,t-1] - m.generation[g,t] <= generators_dict[g]["max_power_mw"] * (1/generators_dict[g]["total_power_hours"])

    # Maximum reserve restriction
    @m.Constraint(m.GENERATORS, m.T)
    def gen_max_fast_reserve(m, g, t):
        return m.generation_fast_reserve[g,t] <= generators_dict[g]["max_fast_reserve_mw"]

    @m.Constraint(m.GENERATORS, m.T)
    def gen_max_slow_reserve(m, g, t):
        return m.generation_fast_reserve[g,t] +  m.generation_slow_reserve[g,t] <= generators_dict[g]["max_slow_reserve_mw"]


    # Maximum capacity restriction
    @m.Constraint(m.GENERATORS, m.T)
    def gen_max_capacity(m, g, t):
        return m.generation[g,t] + m.generation_fast_reserve[g,t] + m.generation_slow_reserve[g,t] <= generators_dict[g]["max_power_mw"] * m.generation_is_dispatched[g,t]


    # Wind generation constraints-----------------
    #
    # Maximum power output
    @m.Constraint(m.WIND_GENERATORS, m.T)
    def wind_gen_max_power(m, w, t):
        if wind_generation.loc[t][w] == 0:
            m.wind_generation[w,t] == 0
        else:
            return m.wind_generation[w,t] == wind_generation.loc[t][w] * m.wind_generation_percentage[w,t]

    # Solar generation constraints-----------------
    #
    # Maximum power output
    @m.Constraint(m.SOLAR_GENERATORS, m.T)
    def solar_gen_max_power(m, spv, t):
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

    # Energy storage for next time period considering efficiency
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
        
    # Storage reserve constraints-----------------
    # Maximum upward power capacity
    @m.Constraint(m.STORAGE, m.T)
    def storage_max_reserve_available(m, s, t):
        return m.storage_discharge_power[s,t] - m.storage_charge_power[s,t] + m.storage_fast_reserve_capacity[s,t] <= storage_dict[s]["max_power_mw"]

    # Maximum next-interval energy storage
    @m.Constraint(m.STORAGE, m.T)
    def storage_min_energy_available(m, s, t):
        if t == 1:
            return pyo.Constraint.Skip 
        eff = storage_dict[s]["efficiency"]
        
        return storage_dict[s]["min_capacity_mwh"] <= m.storage_energy[s,t-1] + ((eff * m.storage_charge_power[s,t]) - ((m.storage_discharge_power[s,t] + m.storage_fast_reserve_capacity[s,t])/eff))*interval_length

    # check the whole model
    # m.pprint()
    # raise

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

    marginal_price_fast_reserve = [m.dual[m.fast_reserve_balance[t]]/interval_length for t in m.T]
    marginal_price_fast_reserve_df = pd.DataFrame(zip(m.T, marginal_price_fast_reserve), columns=["Interval", "Marginal Price FR"])
    marginal_price_fast_reserve_df.set_index("Interval", inplace=True)

    marginal_price_power_df["FR"] = marginal_price_fast_reserve_df["Marginal Price FR"]

    marginal_price_slow_reserve = [m.dual[m.slow_reserve_balance[t]]/interval_length for t in m.T]
    marginal_price_slow_reserve_df = pd.DataFrame(zip(m.T, marginal_price_slow_reserve), columns=["Interval", "Marginal Price SR"])
    marginal_price_slow_reserve_df.set_index("Interval", inplace=True)

    marginal_price_power_df["SR"] = marginal_price_slow_reserve_df["Marginal Price SR"]

    utils.write_temp_csv_file(df=marginal_price_power_df, file_name="price_forecast")
    # print(marginal_price_power_df)
    # m.storage_energy.pprint()
    # m.storage_max_energy_capacity.pprint()
    # # plot marginal price




    # lineplot of generation versus time

    # print("\n\n")
    storage_energy = pd.Series(m.storage_energy.get_values()).unstack(0)
    storage_charge_power = pd.Series(m.storage_charge_power.get_values()).unstack(0)
    storage_discharge_power = pd.Series(m.storage_discharge_power.get_values()).unstack(0)

    gen = pd.Series(m.generation.get_values()).unstack(0)
    wind_gen = pd.Series(m.wind_generation.get_values()).unstack(0)
    solar_gen = pd.Series(m.solar_generation.get_values()).unstack(0)

    df = pd.concat([gen, wind_gen, solar_gen, -storage_charge_power, storage_discharge_power], axis=1)
    fig, axs = plt.subplots(2, 3)
    df.plot(kind='line', ax=axs[1,0])
    df.plot(kind='area', stacked=True, ax=axs[1,1])

    storage_energy.plot(ax=axs[1,2], label='Storage Energy')
    storage_charge_power.plot(ax=axs[1,2], label='Charge Power')
    storage_discharge_power.plot(ax=axs[1,2], label='Discharge Power')


    marginal_price_power_df.plot(kind='line', ax=axs[0,0])
    # marginal_price_power_df.plot(kind='line', ax=axs[0,2])


    # print("\n\n")
    storage_fast_reserve = pd.Series(m.storage_fast_reserve_capacity.get_values()).unstack(0)
    gen_fast_reserve = pd.Series(m.generation_fast_reserve.get_values()).unstack(0)

    df = pd.concat([gen_fast_reserve, storage_fast_reserve], axis=1)
    df.plot(kind='area', stacked=True, ax=axs[0,1])

    # print("\n\n")
    gen_slow_reserve = pd.Series(m.generation_slow_reserve.get_values()).unstack(0)

    df = pd.concat([gen_slow_reserve], axis=1)
    df.plot(kind='area', stacked=True, ax=axs[0,2])


    if __name__ == "__main__":
        plt.show()

    # Calculate the total cost of the system
    total_cost = m.obj()
    print(f"Total cost: {total_cost:,f}")
    return total_cost
    

if __name__ == "__main__":
    main()