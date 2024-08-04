import pyomo.environ as pyo
from pyomo.opt import SolverFactory, TerminationCondition
import pandas as pd
import utils
import os
import matplotlib.pyplot as plt


interval_length = 0.5 # half an hour
interval_num = 48 # 48 periods of 30 minutes

def main():
    
    print("---------Runing Co-optimised Auction-----------")
    solver = pyo.SolverFactory("gurobi")

    m = pyo.ConcreteModel()
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    # Trading intervals, 48 30 min intervals
    m.T = pyo.RangeSet(interval_num)

    # 30 minute Demand data
    demand = utils.get_demand_data()

    solar_generation = utils.get_solar_generator_data()
    wind_generation = utils.get_wind_generator_data()

    max_renewable_generation = [w+s for w,s, in zip(wind_generation.sum(axis=1).to_list(), solar_generation.sum(axis=1).to_list())]
    
    # Reserve demand data
    fast_reserve_demand = [20 + d*0.05 + r*0.05 for d,r in zip(demand, max_renewable_generation)]
    slow_reserve_demand = [150 + d*0.05 + r*0.05 for d,r in zip(demand, max_renewable_generation)]


    # Generators data--------------------------------------------------------------
    generators_dict = utils.get_generator_data()
    generators_cost_dict = utils.get_linearised_conventional_generator_costs()
    num_generation_segments = 10

    m.GENERATORS = pyo.Set(initialize=generators_dict.keys())
    m.gen_segments = pyo.RangeSet(num_generation_segments)

    m.generation = pyo.Var(m.GENERATORS, m.T, domain=pyo.NonNegativeReals)
    m.generation_segments = pyo.Var(m.GENERATORS, m.T, m.gen_segments, domain=pyo.NonNegativeReals)
    
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
    
    energy_cost_wind_generators = sum(wind_generation_var_cost * m.wind_generation[k,t] * interval_length for k in m.WIND_GENERATORS for t in m.T)
    energy_cost_solar_generators = sum(solar_generation_var_cost * m.solar_generation[k,t] * interval_length for k in m.SOLAR_GENERATORS for t in m.T)
    energy_cost_storage_charge = -sum(storage_dict[k]["charge_price"] * m.storage_charge_power[k,t] * interval_length for k in m.STORAGE for t in m.T)
    energy_cost_storage_discharge = sum(storage_dict[k]["discharge_price"] * m.storage_discharge_power[k,t] * interval_length for k in m.STORAGE for t in m.T)

    fast_reserve_cost_generators = sum(generators_cost_dict[k][10]["cost_per_mwh"] * 0.02 * m.generation_fast_reserve[k,t] * interval_length for k in m.GENERATORS for t in m.T)
    fast_reserve_cost_storage = sum(storage_dict[k]["fast_reserve_price"] * m.storage_fast_reserve_capacity[k,t] * interval_length for k in m.STORAGE for t in m.T)

    slow_reserve_cost_generators = sum(generators_cost_dict[k][10]["cost_per_mwh"] * 0.018 * m.generation_slow_reserve[k,t] * interval_length for k in m.GENERATORS for t in m.T)


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

    # Generation total generation equal to sum of segments constraint
    @m.Constraint(m.GENERATORS, m.T)
    def gen_total_segments_power(m, g, t):        
        return m.generation[g,t] == generators_dict[g]["min_power_mw"] * m.generation_is_dispatched[g,t] + sum(m.generation_segments[g,t,gs] for gs in m.gen_segments)

    # Generation segments constrained to their limits
    @m.Constraint(m.GENERATORS, m.T, m.gen_segments)
    def gen_each_segment_power(m, g, t, gs):
        return m.generation_segments[g,t,gs] <= generators_cost_dict[g][gs]["capacity_mw"] * m.generation_is_dispatched[g,t]

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
        if wind_generation.loc[t][w] == 0:
            return m.wind_generation[w,t] == 0
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
        return m.storage_energy[s,interval_num] == soc_end_of_day * storage_dict[s]["max_capacity_mwh"]
        
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

    utils.write_temp_price_file(df=marginal_price_power_df, auction_type="cooptimised")
    # print(marginal_price_power_df)
    # m.storage_energy.pprint()
    # m.storage_max_energy_capacity.pprint()
    # # plot marginal price


    # Assembly results in a dictionary
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
            
            model_results_dict["gen_units"][g][t]["fast_reserve"] = m.generation_fast_reserve[g,t].value
            model_results_dict["gen_units"][g][t]["slow_reserve"] = m.generation_slow_reserve[g,t].value


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
            model_results_dict["gen_units"][s][t]["fast_reserve"] = m.storage_fast_reserve_capacity[s,t].value

    # Power prices
    marginal_price_power = [m.dual[m.power_balance[t]]/interval_length for t in m.T]
    model_results_dict["main_results"]["marginal_price"] = {}
    for t in m.T:
        model_results_dict["main_results"]["marginal_price"][t] = {"power": marginal_price_power[t-1]}
        model_results_dict["main_results"]["marginal_price"][t]["FR"] = marginal_price_fast_reserve[t-1]
        model_results_dict["main_results"]["marginal_price"][t]["SR"] = marginal_price_slow_reserve[t-1]
    
    # System cost
    model_results_dict["main_results"]["system_cost"] = m.obj()

    
    model_results_dict["main_results"]["demand"] = {}
    for t in m.T:
        model_results_dict["main_results"]["demand"][t] = {"power": demand[t-1], "fast": fast_reserve_demand[t-1], "slow": slow_reserve_demand[t-1]}
                

    # Export results to a json file
    print("Exporting results to json file")
    utils.export_dict_to_temp_json(dict_data=model_results_dict, file_name="cooptimised_auction_results")  




    # lineplot of generation versus time
    if __name__ == "__main__":

        # print("\n\n")
        storage_energy = pd.Series(m.storage_energy.get_values()).unstack(0)
        storage_charge_power = pd.Series(m.storage_charge_power.get_values()).unstack(0)
        storage_discharge_power = pd.Series(m.storage_discharge_power.get_values()).unstack(0)

        gen = pd.Series(m.generation.get_values()).unstack(0)
        wind_gen = pd.Series(m.wind_generation.get_values()).unstack(0)
        solar_gen = pd.Series(m.solar_generation.get_values()).unstack(0)

        df = pd.concat([gen, wind_gen, solar_gen, -storage_charge_power, storage_discharge_power], axis=1)
        fig, axs = plt.subplots(2, 3)
        # subplots title
        fig.suptitle('Co-optimised Auction Results')
        df.plot(kind='line', ax=axs[1,0])
        df.plot(kind='area', stacked=True, ax=axs[1,1])

        # Add title to graph and axis
        axs[1,0].set_title('Generation')
        axs[1,0].set_xlabel('Period')
        axs[1,0].set_ylabel('Power [MW]')
        # axs[1,0].legend(loc='center left', bbox_to_anchor=(0.5, 0.5))

        axs[1,1].set_title('Generation')
        axs[1,1].set_xlabel('Period')
        axs[1,1].set_ylabel('Power [MW]')
        # axs[1,1].legend(loc='center left', bbox_to_anchor=(0.5, 0.5))

        storage_energy.plot(ax=axs[1,2], label='Storage Energy')
        storage_charge_power.plot(ax=axs[1,2], label='Charge Power')
        storage_discharge_power.plot(ax=axs[1,2], label='Discharge Power')
        axs[1,2].set_title('Storage')
        axs[1,2].set_xlabel('Period')
        axs[1,2].set_ylabel('Power [MW]')
        # axs[1,2].legend(loc='center left', bbox_to_anchor=(0.5, 0.5))



        marginal_price_power_df.plot(kind='line', ax=axs[0,0])
        # marginal_price_power_df.plot(kind='line', ax=axs[0,2])

        axs[0,0].set_title('Marginal Price')
        axs[0,0].set_xlabel('Period')
        axs[0,0].set_ylabel('Price [$/MWh]')
        # axs[0,0].legend(loc='center left', bbox_to_anchor=(0.5, 0.5))

        storage_fast_reserve = pd.Series(m.storage_fast_reserve_capacity.get_values()).unstack(0)
        gen_fast_reserve = pd.Series(m.generation_fast_reserve.get_values()).unstack(0)

        df = pd.concat([gen_fast_reserve, storage_fast_reserve], axis=1)
        df.plot(kind='area', stacked=True, ax=axs[0,1])

        axs[0,1].set_title('Fast Reserve')
        axs[0,1].set_xlabel('Period')
        axs[0,1].set_ylabel('Power [MW]')
        # axs[0,1].legend(loc='center left', bbox_to_anchor=(0.5, 0.5))

        gen_slow_reserve = pd.Series(m.generation_slow_reserve.get_values()).unstack(0)
        df = pd.concat([gen_slow_reserve], axis=1)
        df.plot(kind='area', stacked=True, ax=axs[0,2])
        axs[0,2].set_title('Slow Reserve')
        axs[0,2].set_xlabel('Period')
        axs[0,2].set_ylabel('Power [MW]')
            

        plt.show()


    print("\n---------Cooptimised auction finished---------\n\n")
    

if __name__ == "__main__":
    main()