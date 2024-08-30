import Cooptimised_auction
import Energy_only_auction
import Simultaneous_auction
import Sequential_auction
import utils
import json
import os
import pandas as pd

utils.linearise_conventional_generator_costs()
auctions = ["cooptimised","simultaneous","sequential"]


def run_wind():

    max_wind_generation = list(range(300, 2301, 200))
    max_solar_generation = [200 for _ in max_wind_generation]

    auction_summary = {e:{a:{} for a in auctions} for e in max_wind_generation}

    final_results = ""

    for wind_p, solar_p in zip(max_wind_generation, max_solar_generation):
        print("New run with wind power: ", wind_p, " and solar power: ", solar_p)

        final_results += "\n\n" + f"max wind = {wind_p:,.0f}MW, solar = {solar_p:,.0f}MW" + "\n--------\n"
        utils.set_renewable_generation_data("solar", solar_p)
        utils.set_renewable_generation_data("wind", wind_p)

        # Total renewable generation before curltailment
        renewable_generation_gross = sum(utils.get_renewable_generation_data_gross("solar"))
        renewable_generation_gross += sum(utils.get_renewable_generation_data_gross("wind"))

        Cooptimised_auction.main()
        Energy_only_auction.main()
        Simultaneous_auction.main()
        Sequential_auction.main()


        # Final result extraction and comparisson
        for auction in ["cooptimised","simultaneous","sequential"]:
            total_cost, power_cost, fr_cost, sr_cost, uplift_cost = utils.get_total_system_cost(auction)
            real_cost = utils.get_real_operation_costs(auction)
            data = utils.get_result_json(auction)

            power_demand = sum(data["main_results"]["demand"][str(t)]["power"] for t in range(1, 49))
            renewable_generation_net = 0

            for g,v in data["gen_units"].items():
                if v["g_type"] not in ["solar","wind"]:
                    continue
                renewable_generation_net += sum([v[str(t)]["generation"] for t in range(1, 49)])

            print(f"Power demand: {power_demand:,.0f}MWh")
            print(f"Renewable generation net: {renewable_generation_net:,.0f}MWh")

            if auction == "cooptimised":
                cooptimised_results = data
                cooptimised_results_2 = total_cost
                text = f"""{auction} auction system cost: {data["main_results"]["system_cost"]:,.0f} ---- {total_cost:,.0f} ---- {renewable_generation_net*100/power_demand:,.1f}% renewable"""
                final_results += text + "\n"
            else:
                change = (data["main_results"]["system_cost"] - cooptimised_results["main_results"]["system_cost"]) / cooptimised_results["main_results"]["system_cost"] * 100
                change_2 = ((total_cost - cooptimised_results_2) / cooptimised_results_2) * 100
                text = f"""{auction} auction system cost: {data["main_results"]["system_cost"]:,.0f} +{change:.2f}% ---- {total_cost:,.0f} +{change_2:.2f}% ---- {renewable_generation_net*100/power_demand:,.1f}% renewable"""
                final_results += text + "\n"
                
                # print(f"""{auction} auction system cost: {data["main_results"]["system_cost"]:,.0f} +{change:.2f}%""")

            auction_summary[wind_p][auction] = {}
            auction_summary[wind_p][auction]["type"] = "Varying wind power"
            auction_summary[wind_p][auction]["system_cost"] = total_cost
            auction_summary[wind_p][auction]["power_cost"] = power_cost
            auction_summary[wind_p][auction]["fr_cost"] = fr_cost
            auction_summary[wind_p][auction]["sr_cost"] = sr_cost
            auction_summary[wind_p][auction]["uplift_cost"] = uplift_cost
            auction_summary[wind_p][auction]["real_operation_cost_no_increment"] = real_cost

            auction_summary[wind_p][auction]["model_obj_func"] = data["main_results"]["system_cost"]

            auction_summary[wind_p][auction]["renewables_net"] = renewable_generation_net*100/power_demand
            auction_summary[wind_p][auction]["renewables_gross"] = renewable_generation_gross*100/power_demand


    print("\n\n\n--------------FINAL RESULTS-------------\n\n")
    print(final_results)


    #  save auction_summary to a file
    with open(os.path.join(os.getcwd(), "bin", "temp","auction_summary.json"), "w") as f:
        json.dump(auction_summary, f)


def run_conventional_reserve():

    reserve_price_increases = [1*i for i in range(0,13,2)]
    # reserve_price_increases = [3]

    auction_summary = {f"{e*100:,.0f}%":{a:{} for a in auctions} for e in reserve_price_increases}

    # Set renewable level
    utils.set_renewable_generation_data("solar", 200)
    utils.set_renewable_generation_data("wind", 700)

    # Total renewable generation before curltailment
    renewable_generation_gross = sum(utils.get_renewable_generation_data_gross("solar"))
    renewable_generation_gross += sum(utils.get_renewable_generation_data_gross("wind"))

    # final_results = ""
    for r_p_i in reserve_price_increases:
        print(f"New run reserve price increase of, {r_p_i*100:,.0f}%")

        # final_results += "\n\n" + f"max wind = {wind_p:,.0f}MW, solar = {solar_p:,.0f}MW" + "\n--------\n"

        params = {"reserve_price_inc": r_p_i}

        Cooptimised_auction.main(params=params)
        Energy_only_auction.main(params=params)
        Simultaneous_auction.main(params=params)
        Sequential_auction.main(params=params)


        # Final result extraction and comparisson
        for auction in ["cooptimised","simultaneous","sequential"]:
            total_cost, power_cost, fr_cost, sr_cost, uplift_cost = utils.get_total_system_cost(auction, params)
            real_cost = utils.get_real_operation_costs(auction)
            data = utils.get_result_json(auction)

            power_demand = sum(data["main_results"]["demand"][str(t)]["power"] for t in range(1, 49))
            renewable_generation_net = 0

            for g,v in data["gen_units"].items():
                if v["g_type"] not in ["solar","wind"]:
                    continue
                renewable_generation_net += sum([v[str(t)]["generation"] for t in range(1, 49)])

            print(f"Power demand: {power_demand:,.0f}MWh")
            print(f"Renewable generation net: {renewable_generation_net:,.0f}MWh")

            auction_summary[f"{r_p_i*100:,.0f}%"][auction] = {}
            auction_summary[f"{r_p_i*100:,.0f}%"][auction]["type"] = "Varying conventional reserve price"
            auction_summary[f"{r_p_i*100:,.0f}%"][auction]["system_cost"] = total_cost
            auction_summary[f"{r_p_i*100:,.0f}%"][auction]["power_cost"] = power_cost
            auction_summary[f"{r_p_i*100:,.0f}%"][auction]["fr_cost"] = fr_cost
            auction_summary[f"{r_p_i*100:,.0f}%"][auction]["sr_cost"] = sr_cost
            auction_summary[f"{r_p_i*100:,.0f}%"][auction]["uplift_cost"] = uplift_cost
            auction_summary[f"{r_p_i*100:,.0f}%"][auction]["real_operation_cost_no_increment"] = real_cost

            auction_summary[f"{r_p_i*100:,.0f}%"][auction]["model_obj_func"] = data["main_results"]["system_cost"]

            auction_summary[f"{r_p_i*100:,.0f}%"][auction]["renewables_net"] = renewable_generation_net*100/power_demand
            auction_summary[f"{r_p_i*100:,.0f}%"][auction]["renewables_gross"] = renewable_generation_gross*100/power_demand


    print("\n\n\n--------------FINAL RESULTS-------------\n\n")
    # print(final_results)


    #  save auction_summary to a file
    with open(os.path.join(os.getcwd(), "bin", "temp","auction_summary.json"), "w") as f:
        json.dump(auction_summary, f)


def run_storage_reserve():

    reserve_price_increases = [0.4+(i*0.1) for i in range(6,-1,-1)]
    # reserve_price_increases = [3]

    auction_summary = {f"{e*100:,.0f}%":{a:{} for a in auctions} for e in reserve_price_increases}

    # Set renewable level
    utils.set_renewable_generation_data("solar", 200)
    utils.set_renewable_generation_data("wind", 1700)

    # Total renewable generation before curltailment
    renewable_generation_gross = sum(utils.get_renewable_generation_data_gross("solar"))
    renewable_generation_gross += sum(utils.get_renewable_generation_data_gross("wind"))

    # final_results = ""
    for r_p_i in reserve_price_increases:
        print(f"New run reserve price increase of, {r_p_i*100:,.0f}%")

        # final_results += "\n\n" + f"max wind = {wind_p:,.0f}MW, solar = {solar_p:,.0f}MW" + "\n--------\n"

        params = {"storage_reserve_price_inc": r_p_i}

        Cooptimised_auction.main(params=params)
        Energy_only_auction.main(params=params)
        Simultaneous_auction.main(params=params)
        Sequential_auction.main(params=params)


        # Final result extraction and comparisson
        for auction in ["cooptimised","simultaneous","sequential"]:
            total_cost, power_cost, fr_cost, sr_cost, uplift_cost = utils.get_total_system_cost(auction, params)
            real_cost = utils.get_real_operation_costs(auction)
            data = utils.get_result_json(auction)

            power_demand = sum(data["main_results"]["demand"][str(t)]["power"] for t in range(1, 49))
            renewable_generation_net = 0

            for g,v in data["gen_units"].items():
                if v["g_type"] not in ["solar","wind"]:
                    continue
                renewable_generation_net += sum([v[str(t)]["generation"] for t in range(1, 49)])

            print(f"Power demand: {power_demand:,.0f}MWh")
            print(f"Renewable generation net: {renewable_generation_net:,.0f}MWh")


            auction_summary[f"{r_p_i*100:,.0f}%"][auction] = {}
            auction_summary[f"{r_p_i*100:,.0f}%"][auction]["type"] = "Varying storage reserve price"
            auction_summary[f"{r_p_i*100:,.0f}%"][auction]["system_cost"] = total_cost
            auction_summary[f"{r_p_i*100:,.0f}%"][auction]["power_cost"] = power_cost
            auction_summary[f"{r_p_i*100:,.0f}%"][auction]["fr_cost"] = fr_cost
            auction_summary[f"{r_p_i*100:,.0f}%"][auction]["sr_cost"] = sr_cost
            auction_summary[f"{r_p_i*100:,.0f}%"][auction]["uplift_cost"] = uplift_cost
            auction_summary[f"{r_p_i*100:,.0f}%"][auction]["real_operation_cost_no_increment"] = real_cost
            
            auction_summary[f"{r_p_i*100:,.0f}%"][auction]["model_obj_func"] = data["main_results"]["system_cost"]

            auction_summary[f"{r_p_i*100:,.0f}%"][auction]["renewables_net"] = renewable_generation_net*100/power_demand
            auction_summary[f"{r_p_i*100:,.0f}%"][auction]["renewables_gross"] = renewable_generation_gross*100/power_demand


    print("\n\n\n--------------FINAL RESULTS-------------\n\n")
    # print(final_results)


    #  save auction_summary to a file
    with open(os.path.join(os.getcwd(), "bin", "temp","auction_summary.json"), "w") as f:
        json.dump(auction_summary, f)


def make_graphs():
    # Load saved json file into dict
    with open(os.path.join(os.getcwd(), "bin", "temp","auction_summary.json"), "r") as f:
        auction_summary = json.load(f)

    scenarios = list(auction_summary.keys())

    print("\n\n\n--------------GRAPHING-------------\n\n")
    # Barchart with the results
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams["font.family"] = "Century"

    figA, ax1 = plt.subplots()

    bar_width = 0.25
    opacity = 0.8

    index = np.arange(len(scenarios))

    for i, auction in enumerate(auctions):
        title = [auction_summary[s][auction]["type"] for s in scenarios]
        title = title[0]

        costs = [auction_summary[s][auction]["system_cost"] for s in scenarios]
        if auction == "cooptimised":
            cooptimised_costs = costs

        ax1.bar(index + i * bar_width, costs, bar_width, alpha=opacity, label=auction)        
        rects = ax1.patches
        # Make some labels.

        for rect, cost, ii in zip(rects, costs, range(len(costs))):
            # Include conditional to only do for not cooptimised, and add only % difference
            if auction == "cooptimised":
                continue
            
            height = cost
            label = f"{((cost - cooptimised_costs[ii])/cooptimised_costs[ii])*100:+.1f}%"
            # print(label)
            ax1.text(
                i*bar_width + rect.get_x() + rect.get_width() / 2, height + 5, label, ha="center", va="bottom", fontdict={"size":"x-small"}
            )
        
        ax1.set_xlabel("Scenario")
        ax1.set_ylabel("System cost")
        ax1.set_title(f"System cost for different auction types - {title}")
        ax1.set_xticks(index + bar_width)
        ax1.set_xticklabels([f"{s}MW" for s in scenarios])
        ax1.legend()


    renewables_net = [auction_summary[s][auctions[0]]["renewables_net"] for s in scenarios]
    renewables_gross = [auction_summary[s][auctions[0]]["renewables_gross"] for s in scenarios]
    # ax1.plot(index + 1 * bar_width, renewables_net,color="r", linewidth=1.0)
    # plt.xlim([-bar_width, len(index)-bar_width])

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

    color = 'k'
    ax2.set_ylabel(r"% of renewables")#, color=color)  # we already handled the x-label with ax1
    ax2.plot(index+bar_width, renewables_net, color=color, linestyle='-', label="Net")
    ax2.plot(index+bar_width, renewables_gross, color="r", linestyle='--', label="Gross")
    ax2.legend()
    ax2.tick_params(axis='y', labelcolor=color)
    # ax2.set(ylim=(0, 80))
    # fig.tight_layout()  # otherwise the right y-label is slightly clipped

    # plt.show()
    
# New chart version
    figB, ax3 = plt.subplots()

    for i, auction in enumerate(auctions):
        costs = [auction_summary[s][auction]["system_cost"] for s in scenarios]
        power_cost = [auction_summary[s][auction]["power_cost"] for s in scenarios]
        fr_cost = [auction_summary[s][auction]["fr_cost"] for s in scenarios]
        sr_cost = [auction_summary[s][auction]["sr_cost"] for s in scenarios]
        uplift_cost = [auction_summary[s][auction]["uplift_cost"] for s in scenarios]

        cost_no_uplift = [p+fr+sr for p, fr, sr in zip(power_cost, fr_cost, sr_cost)]

        if auction == "cooptimised":
            cooptimised_costs = costs

        ax3.bar(index + i * bar_width, cost_no_uplift, width=bar_width, label=f'{auction} Market costs', color="blue")
        ax3.bar(index + i * bar_width, uplift_cost, width=bar_width, bottom=cost_no_uplift, label=f'{auction} Uplift costs', color="red")
        
        # ax3.bar(index + i * bar_width, costs, bar_width, alpha=opacity, label=auction)        
        rects = ax3.patches
        # Make some labels.

        for rect, cost, ii in zip(rects, costs, range(len(costs))):
            # Include conditional to only do for not cooptimised, and add only % difference
            if auction == "cooptimised":
                continue
            
            height = cost
            label = f"{((cost - cooptimised_costs[ii])/cooptimised_costs[ii])*100:+.1f}%"
            # print(label)
            ax3.text(
                i*bar_width + rect.get_x() + rect.get_width() / 2, height + 5, label, ha="center", va="bottom", fontdict={"size":"x-small"}
            )
        
        ax3.set_xlabel("Scenario")
        ax3.set_ylabel("System cost")
        ax3.set_title(f"System cost for different auction types - {title}")
        ax3.set_xticks(index + bar_width)
        ax3.set_xticklabels([f"{s}MW" for s in scenarios])
        ax3.legend()
        # Assuming you have two figures, fig1 and fig2, each with an axis ax1 and ax2 respectively.

    renewables_net = [auction_summary[s][auctions[0]]["renewables_net"] for s in scenarios]
    renewables_gross = [auction_summary[s][auctions[0]]["renewables_gross"] for s in scenarios]
    # ax3.plot(index + 1 * bar_width, renewables_net,color="r", linewidth=1.0)
    # plt.xlim([-bar_width, len(index)-bar_width])

    ax4 = ax3.twinx()  # instantiate a second Axes that shares the same x-axis

    color = 'k'
    ax4.set_ylabel(r"% of renewables")#, color=color)  # we already handled the x-label with ax1
    ax4.plot(index+bar_width, renewables_net, color=color, linestyle='-', label="Net")
    ax4.plot(index+bar_width, renewables_gross, color="r", linestyle='--', label="Gross")
    ax4.legend()
    ax4.tick_params(axis='y', labelcolor=color)
    # ax4.set(ylim=(0, 80))
    # fig.tight_layout()  # otherwise the right y-label is slightly clipped

    # plt.show()

# New chart version
    figC, ax5 = plt.subplots()

    for i, auction in enumerate(auctions):
        costs = [auction_summary[s][auction]["system_cost"] for s in scenarios]
        power_cost = [auction_summary[s][auction]["power_cost"] for s in scenarios]
        fr_cost = [auction_summary[s][auction]["fr_cost"] for s in scenarios]
        sr_cost = [auction_summary[s][auction]["sr_cost"] for s in scenarios]
        uplift_cost = [auction_summary[s][auction]["uplift_cost"] for s in scenarios]

        if auction == "cooptimised":
            cooptimised_costs = costs

        ax5.bar(index + i * bar_width, power_cost, width=bar_width, label=f'{auction} Power', color="green")
        ax5.bar(index + i * bar_width, fr_cost, width=bar_width, bottom=power_cost, label=f'{auction} FR', color="orange")
        ax5.bar(index + i * bar_width, sr_cost, width=bar_width, bottom=[p+fr for p, fr in zip(power_cost, fr_cost)], label=f'{auction} SR', color="purple")
        ax5.bar(index + i * bar_width, uplift_cost, width=bar_width, bottom=[p+fr+sr for p, fr, sr in zip(power_cost, fr_cost, sr_cost)], label=f'{auction} Uplift', color="red")

        # ax5.bar(index + i * bar_width, costs, bar_width, alpha=opacity, label=auction)        
        rects = ax5.patches
        # Make some labels.

        for rect, cost, ii in zip(rects, costs, range(len(costs))):
            # Include conditional to only do for not cooptimised, and add only % difference
            if auction == "cooptimised":
                continue
            
            height = cost
            label = f"{((cost - cooptimised_costs[ii])/cooptimised_costs[ii])*100:+.1f}%"
            # print(label)
            ax5.text(
                i*bar_width + rect.get_x() + rect.get_width() / 2, height + 5, label, ha="center", va="bottom", fontdict={"size":"x-small"}
            )
        
        ax5.set_xlabel("Scenario")
        ax5.set_ylabel("System cost")
        ax5.set_title(f"System cost for different auction types - {title}")
        ax5.set_xticks(index + bar_width)
        ax5.set_xticklabels([f"{s}MW" for s in scenarios])
        ax5.legend()


    renewables_net = [auction_summary[s][auctions[0]]["renewables_net"] for s in scenarios]
    renewables_gross = [auction_summary[s][auctions[0]]["renewables_gross"] for s in scenarios]
    # ax5.plot(index + 1 * bar_width, renewables_net,color="r", linewidth=1.0)
    # plt.xlim([-bar_width, len(index)-bar_width])

    ax6 = ax5.twinx()  # instantiate a second Axes that shares the same x-axis

    color = 'k'
    ax6.set_ylabel(r"% of renewables")#, color=color)  # we already handled the x-label with ax1
    ax6.plot(index+bar_width, renewables_net, color=color, linestyle='-', label="Net")
    ax6.plot(index+bar_width, renewables_gross, color="r", linestyle='--', label="Gross")
    ax6.legend()
    ax6.tick_params(axis='y', labelcolor=color)
    # ax6.set(ylim=(0, 80))
    # fig.tight_layout()  # otherwise the right y-label is slightly clipped

    # Copy x-axis and y-axis limits
    ax3.set_ylim(ax1.get_ylim())
    ax5.set_ylim(ax1.get_ylim())
    
    figD, ax7 = plt.subplots()

    for i, auction in enumerate(auctions):
        costs = [auction_summary[s][auction]["system_cost"] for s in scenarios]
        model_obj = [auction_summary[s][auction]["model_obj_func"] for s in scenarios]

        gen_surplus = [c-m for m, c in zip(model_obj, costs)]

        if auction == "cooptimised":
            cooptimised_costs = costs

        ax7.bar(index + i * bar_width, model_obj, width=bar_width, label=f'{auction} Obj Func', color="green")
        ax7.bar(index + i * bar_width, gen_surplus, width=bar_width, bottom=model_obj, label=f'{auction} gen surplus', color="orange")
        
        # ax7.bar(index + i * bar_width, costs, bar_width, alpha=opacity, label=auction)        
        rects = ax7.patches
        # Make some labels.

        for rect, cost, ii in zip(rects, costs, range(len(costs))):
            # Include conditional to only do for not cooptimised, and add only % difference
            if auction == "cooptimised":
                continue
            
            height = cost
            label = f"{((cost - cooptimised_costs[ii])/cooptimised_costs[ii])*100:+.1f}%"
            # print(label)
            ax7.text(
                i*bar_width + rect.get_x() + rect.get_width() / 2, height + 5, label, ha="center", va="bottom", fontdict={"size":"x-small"}
            )
        
        ax7.set_xlabel("Scenario")
        ax7.set_ylabel("System cost")
        ax7.set_title(f"System cost for different auction types - {title}")
        ax7.set_xticks(index + bar_width)
        ax7.set_xticklabels([f"{s}MW" for s in scenarios])
        ax7.legend()


    renewables_net = [auction_summary[s][auctions[0]]["renewables_net"] for s in scenarios]
    renewables_gross = [auction_summary[s][auctions[0]]["renewables_gross"] for s in scenarios]
    # ax7.plot(index + 1 * bar_width, renewables_net,color="r", linewidth=1.0)
    # plt.xlim([-bar_width, len(index)-bar_width])

    ax8 = ax7.twinx()  # instantiate a second Axes that shares the same x-axis

    color = 'k'
    ax8.set_ylabel(r"% of renewables")#, color=color)  # we already handled the x-label with ax1
    ax8.plot(index+bar_width, renewables_net, color=color, linestyle='-', label="Net")
    ax8.plot(index+bar_width, renewables_gross, color="r", linestyle='--', label="Gross")
    ax8.legend()
    ax8.tick_params(axis='y', labelcolor=color)
    # ax8.set(ylim=(0, 80))
    # fig.tight_layout()  # otherwise the right y-label is slightly clipped

    # Copy x-axis and y-axis limits
    ax3.set_ylim(ax1.get_ylim())
    ax5.set_ylim(ax1.get_ylim())
    ax7.set_ylim(ax1.get_ylim())

    figA.set_size_inches(16, 9)
    figB.set_size_inches(16, 9)
    figC.set_size_inches(16, 9)
    figD.set_size_inches(16, 9)

    figA.savefig(os.path.join(os.getcwd(), "bin", "temp","Fig-A.png"), dpi=300, bbox_inches="tight")
    figB.savefig(os.path.join(os.getcwd(), "bin", "temp","Fig-B.png"), dpi=300, bbox_inches="tight")
    figC.savefig(os.path.join(os.getcwd(), "bin", "temp","Fig-C.png"), dpi=300, bbox_inches="tight")
    figD.savefig(os.path.join(os.getcwd(), "bin", "temp","Fig-D.png"), dpi=300, bbox_inches="tight")


    plt.show()

    # df = pd.read_json(os.path.join(os.getcwd(), "bin", "temp","auction_summary.json"))
    # print(df)
    df = pd.json_normalize(auction_summary).T
    df.columns = ["value"]
    df.reset_index(inplace=True)
    # print(df)
    df[['Scenario', 'Auction', 'Concept']] = df["index"].str.split('.', expand=True)
    df = df[['Scenario', 'Auction', 'Concept', 'value']]
    # Remove rows with string in column value
    df = df[~df["value"].apply(lambda x: isinstance(x, str))]
    print(df)
    df.to_csv(os.path.join(os.getcwd(), "bin", "temp","auction_summary.csv"))


def get_dispatch_info_in_a_nice_way():


    utils.set_renewable_generation_data("solar", 200)
    utils.set_renewable_generation_data("wind", 700)

    # # Total renewable generation before curltailment
    # renewable_generation_gross = sum(utils.get_renewable_generation_data_gross("solar"))
    # renewable_generation_gross += sum(utils.get_renewable_generation_data_gross("wind"))

    Cooptimised_auction.main()
    Energy_only_auction.main()
    Simultaneous_auction.main()
    Sequential_auction.main()


    # Final result extraction and comparisson
    for auction in ["cooptimised", "energy_only","simultaneous","sequential"]:
        
        generation_costs = utils.get_real_operation_costs(auction, separate_reserves=True)
        df_cost = pd.json_normalize(generation_costs).T
        df_cost.columns = ["value"]
        # df_cost = df_cost[~df_cost["value"].apply(lambda x: isinstance(x, str))]
        df_cost.reset_index(inplace=True)
        df_cost[["Unit", "Concept"]] = df_cost["index"].str.split(".", expand=True)

        # column period to int
        # df_cost["Period"] = df_cost["Period"].astype(int)

        df_cost = df_cost[["Unit", "Concept", "value"]]

        # print(df_cost)
        df_cost = df_cost.pivot(index="Unit", columns="Concept", values="value")
        df_cost.to_csv(os.path.join(os.getcwd(), "bin", "temp",f"cost_{auction}.csv"))

        

        data = utils.get_result_json(auction)

        # print(data["gen_units"])

        df_schedule = pd.json_normalize(data["gen_units"]).T
        df_schedule.columns = ["value"]
        df_schedule = df_schedule[~df_schedule["value"].apply(lambda x: isinstance(x, str))]
        df_schedule.reset_index(inplace=True)
        df_schedule[["Unit", "Period", "Concept"]] = df_schedule["index"].str.split(".", expand=True)

        # column period to int
        df_schedule["Period"] = df_schedule["Period"].astype(int)

        df_schedule = df_schedule[["Unit", "Period", "Concept", "value"]]

        # print(df_schedule)
        df_schedule = df_schedule.pivot(index=["Unit", "Period"], columns="Concept", values="value")
        df_schedule.to_csv(os.path.join(os.getcwd(), "bin", "temp",f"schedule_{auction}.csv"))

        

if __name__ == "__main__":


# Uncomment section to be executed

    # Wind scenarios
    run_wind()
    make_graphs()

    # Conventional power plant FCAS price increments
    # run_conventional_reserve()
    # make_graphs()

    # Storage FCAS price increments
    # run_storage_reserve()
    # make_graphs()