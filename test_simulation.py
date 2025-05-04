import os
import traci
import sumolib

def run_simulation():
    step = 0
    for step in range(10000):
        traci.simulationStep()
        step += 1
    traci.close()

if __name__ == "__main__":
    # Update these paths if needed
    sumo_binary = "sumo-gui"  # or "sumo" for non-GUI
    config_file = "sumo_network_16points/simulation.sumocfg"

    sumo_cmd = [sumo_binary, "-c", config_file, "--start"]
    traci.start(sumo_cmd)

    print("Simulation started in GUI...")
    run_simulation()
    print("Simulation completed.")
