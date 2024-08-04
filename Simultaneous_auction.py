import pyomo.environ as pyo
from pyomo.opt import TerminationCondition
import pandas as pd
import utils
import matplotlib.pyplot as plt


interval_length = 0.5 # half an hour
interval_num = 48 # 48 periods of 30 minutes

def main():

    print("---------Runing Simultaneous Reserves Auction-----------")

    # Reserves can only be offered by generators that are dispatched and storage (Fast reserve only)

    solver = pyo.SolverFactory("gurobi")
    m = pyo.ConcreteModel()
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    # Trading intervals, 48 30 min intervals
    m.T = pyo.RangeSet(interval_num)

    # import json with energy auction results and convert to dict
    print("Reading json file")
    energy_auction_results_dict = utils.import_dict_from_temp_json(file_name="energy_only_auction_results")

    # Reserves demand data--------------------------------------------------------------
    # print(energy_auction_results_dict["main_results"])
    
    fast_reserve_demand = [energy_auction_results_dict["main_results"]["demand"][t]["fast"] for t in m.T]
    slow_reserve_demand = [energy_auction_results_dict["main_results"]["demand"][t]["slow"] for t in m.T]

    # Generators data--------------------------------------------------------------
    generators_dict = utils.get_generator_data()
    generators_cost_dict = utils.get_linearised_conventional_generator_costs()
    
    m.GENERATORS = pyo.Set(initialize=generators_dict.keys())
    m.generation_fast_reserve = pyo.Var(m.GENERATORS, m.T, domain=pyo.NonNegativeReals)
    m.generation_slow_reserve = pyo.Var(m.GENERATORS, m.T, domain=pyo.NonNegativeReals)


    # Storage --------------------------------------------------------------
    storage_dict = utils.get_storage_data()
    
    m.STORAGE = pyo.Set(initialize=storage_dict.keys())
    m.storage_fast_reserve_capacity = pyo.Var(m.STORAGE, m.T, domain=pyo.NonNegativeReals)


    # Objective function--------------------------------------------------------------
    fast_reserve_cost_generators = sum(generators_cost_dict[k][10]["cost_per_mwh"] * 0.02 * m.generation_fast_reserve[k,t] * interval_length for k in m.GENERATORS for t in m.T)
    fast_reserve_cost_storage = sum(storage_dict[k]["fast_reserve_price"] * m.storage_fast_reserve_capacity[k,t] * interval_length for k in m.STORAGE for t in m.T)

    slow_reserve_cost_generators = sum(generators_cost_dict[k][10]["cost_per_mwh"] * 0.018 * m.generation_slow_reserve[k,t] * interval_length for k in m.GENERATORS for t in m.T)


    m.obj = pyo.Objective(
        expr = 
            # Energy cost
            + fast_reserve_cost_generators
            + fast_reserve_cost_storage

            + slow_reserve_cost_generators
            
            , sense=pyo.minimize)
    # m.pprint()

    # # Constraints--------------------------------------------------------------

    # Fast reserve fulfilment balance
    @m.Constraint(m.T)
    def fast_reserve_balance(m, t):
        reserved_mw = 0
        reserved_mw += sum(m.generation_fast_reserve[g_name, t] for g_name in m.GENERATORS)
        reserved_mw += sum(m.storage_fast_reserve_capacity[s_name, t] for s_name in m.STORAGE)

        return reserved_mw >= fast_reserve_demand[t - 1]
    # m.fast_reserve_balance.pprint()


    # Slow reserve fulfilment balance
    @m.Constraint(m.T)
    def slow_reserve_balance(m, t):
        reserved_mw = 0
        reserved_mw += sum(m.generation_slow_reserve[g_name, t] for g_name in m.GENERATORS)
        
        return reserved_mw >= slow_reserve_demand[t - 1]
    # m.power_balance.pprint()

    # Generation constraints-----------------
    #

    # Maximum reserve restriction
    @m.Constraint(m.GENERATORS, m.T)
    def gen_max_fast_reserve(m, g, t):
        return m.generation_fast_reserve[g,t] <= generators_dict[g]["max_fast_reserve_mw"]

    @m.Constraint(m.GENERATORS, m.T)
    def gen_max_slow_reserve(m, g, t):
        return m.generation_slow_reserve[g,t] + m.generation_fast_reserve[g,t] <= generators_dict[g]["max_slow_reserve_mw"]



    # Maximum capacity restriction
    @m.Constraint(m.GENERATORS, m.T)
    def gen_max_capacity(m, g, t):
        return m.generation_fast_reserve[g,t] + m.generation_slow_reserve[g,t] <= energy_auction_results_dict["gen_units"][g][t]["available_for_reserves"]

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


    # Calculate the total cost of the system
    total_cost = energy_auction_results_dict["main_results"]["system_cost"] + m.obj()
    print(f"Total cost: {total_cost:,.0f}")
    

    # Marginal price extraction
    marginal_price_power = [v["power"] for _, v in energy_auction_results_dict["main_results"]["marginal_price"].items()]
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

    utils.write_temp_price_file(df=marginal_price_power_df, auction_type="simultaneous")

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
            final_results_dict["gen_units"][g][t]["slow_reserve"] = m.generation_slow_reserve[g,t].value

    # Add reserved capacity for each storage
    for s in m.STORAGE:
        for t in m.T:
            final_results_dict["gen_units"][s][t]["fast_reserve"] = m.storage_fast_reserve_capacity[s,t].value

    # Export results to a json file
    print("Exporting results to json file")
    utils.export_dict_to_temp_json(dict_data=final_results_dict, file_name="simultaneous_auction_results")
    
    # m.fast_reserve_balance.pprint()

    # lineplot of generation versus time
    if __name__ == "__main__":
        print("\n\n")
        storage_energy = pd.Series({(k,t):v[t]["energy"] for t in m.T for k, v in energy_auction_results_dict["gen_units"].items() if v["g_type"] == "storage"}).unstack(0)
        storage_charge_power = pd.Series({(k,t):v[t]["charge"] for t in m.T for k, v in energy_auction_results_dict["gen_units"].items() if v["g_type"] == "storage"}).unstack(0)
        storage_discharge_power = pd.Series({(k,t):v[t]["discharge"] for t in m.T for k, v in energy_auction_results_dict["gen_units"].items() if v["g_type"] == "storage"}).unstack(0)


        gen = pd.Series({(k,t):v[t]["generation"] for t in m.T for k, v in energy_auction_results_dict["gen_units"].items() if v["g_type"] == "conventional"}).unstack(0)
        wind_gen = pd.Series({(k,t):v[t]["generation"] for t in m.T for k, v in energy_auction_results_dict["gen_units"].items() if v["g_type"] == "wind"}).unstack(0)
        solar_gen = pd.Series({(k,t):v[t]["generation"] for t in m.T for k, v in energy_auction_results_dict["gen_units"].items() if v["g_type"] == "solar"}).unstack(0)
        

        df = pd.concat([gen, wind_gen, solar_gen, -storage_charge_power, storage_discharge_power], axis=1)
        fig, axs = plt.subplots(2, 3)
        fig.suptitle('Simultaneous Auction Results')
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

    print("\n---------Simultaneous reserve auction finished---------\n\n")


if __name__ == "__main__":
    main()