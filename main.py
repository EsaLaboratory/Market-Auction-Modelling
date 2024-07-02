import Cooptimised_auction
import Simultaneous_auction

print("---------Runing Co-optimised Auction-----------")

cop_cost = Cooptimised_auction.main()

print("Cooptimised finished")

sim_cost = Simultaneous_auction.main()

print("Simultaneous finished")

print("\n\n\n--------------FINAL RESULTS-------------\n\n")

print(f"Cooptimised cost: {cop_cost:,.0f}")
print(f"Simultaneous cost: {sim_cost:,.0f}")
print(f"Cost difference: {sim_cost - cop_cost:,.0f}")
print(f"Cost difference percentage: {((sim_cost - cop_cost) / sim_cost) * 100:.2f}%\n")
