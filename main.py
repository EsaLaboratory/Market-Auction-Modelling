import Cooptimised_auction
import Energy_only_auction
import Simultaneous_auction
import Sequential_auction
import utils


# Run in case there are changes in the generator's data
utils.linearise_conventional_generator_costs()

# Set different levels of renewable generation and iterate through them
# max_wind_generation = [300]
# max_solar_generation = [200]

graph_data = True

max_wind_generation = list(range(300, 2301, 200))
max_solar_generation = [200 for _ in max_wind_generation]

auctions = ["cooptimised","simultaneous","sequential"]
auction_summary = {e:{a:{} for a in auctions} for e in zip(max_wind_generation, max_solar_generation)}

final_results = ""

for wind_p, solar_p in zip(max_wind_generation, max_solar_generation):
    print("New run with wind power: ", wind_p, " and solar power: ", solar_p)

    final_results += "\n\n" + f"max wind = {wind_p:,.0f}MW, solar = {solar_p:,.0f}MW" + "\n--------\n"
    utils.set_renewable_generation_data("solar", solar_p)
    utils.set_renewable_generation_data("wind", wind_p)

    Cooptimised_auction.main()
    Energy_only_auction.main()
    Simultaneous_auction.main()
    Sequential_auction.main()


    # Final result extraction and comparisson
    for auction in ["cooptimised","simultaneous","sequential"]:
        total_cost = utils.get_total_system_cost(auction)
        data = utils.get_result_json(auction)

        power_demand = sum(data["main_results"]["demand"][str(t)]["power"] for t in range(1, 25))
        renewable_generation_net = 0

        for g,v in data["gen_units"].items():
            if v["g_type"] not in ["solar","wind"]:
                continue
            renewable_generation_net += sum([v[str(t)]["generation"] for t in range(1, 25)])

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

        auction_summary[(wind_p,solar_p)][auction] = {}
        auction_summary[(wind_p,solar_p)][auction]["system_cost"] = total_cost
        auction_summary[(wind_p,solar_p)][auction]["renewables_net"] = renewable_generation_net*100/power_demand


print("\n\n\n--------------FINAL RESULTS-------------\n\n")
print(final_results)

if graph_data:
    print("\n\n\n--------------GRAPHING-------------\n\n")
    # Barchart with the results
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax1 = plt.subplots()

    bar_width = 0.25
    opacity = 0.8

    index = np.arange(len(max_wind_generation))

    for i, auction in enumerate(auctions):
        costs = [auction_summary[(w,s)][auction]["system_cost"] for w,s in zip(max_wind_generation, max_solar_generation)]
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
        ax1.set_title("System cost for different auction types")
        ax1.set_xticks(index + bar_width)
        ax1.set_xticklabels([f"{w}MW, {s}MW" for w,s in zip(max_wind_generation, max_solar_generation)])
        ax1.legend()


    renewables_net = [auction_summary[(w,s)][auctions[0]]["renewables_net"] for w,s in zip(max_wind_generation, max_solar_generation)]
    # ax1.plot(index + 1 * bar_width, renewables_net,color="r", linewidth=1.0)
    # plt.xlim([-bar_width, len(index)-bar_width])

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

    color = 'k'
    ax2.set_ylabel(r"% of renewables")#, color=color)  # we already handled the x-label with ax1
    ax2.plot(index+bar_width, renewables_net, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set(ylim=(0, 80))
    # fig.tight_layout()  # otherwise the right y-label is slightly clipped

    plt.show()


print("---------End-----------")