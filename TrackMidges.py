import numpy as np
import Swarm
import Environment

midgehostratio = 100  # Midge/host ratio

hostpop = 100
midgepop = hostpop * midgehostratio

midges = np.full(midgepop, True)  # ALL FIRST GENERATION MIDGES WILL BE INFECTED, NO OTHERS

hostinf = np.full(hostpop, False)  # Entire host population is naive to BTV

envir = Environment.Envir(length=1000)
host = Swarm.HostSwarm(envir=envir, size=hostpop, infected=hostinf)
swrm = Swarm.MidgeSwarm(envir=envir, size=midgepop, hostswarm=host, infected=midges, dps=1.0, savepositions=True)
swrm.pVtoH = 0  # Don't want to consider transmission to host
swrm.eip = 100  # Again just to be sure
dt = 60  # Step the simulation every 60 seconds (1 minute)
length = 2 * 300  # Total length in days of the simulation

print("Moving swarm...")
# RUN UNTIL ALL INFECTED MIDGES (FIRST GEN) HAVE DIED
for i in range(length):
    swrm.move(dt)

    if i % 300 == 0:
        print("Day", i // 300)

print("Simulation finished")

print("Saving results...")
swrm.writetocsv(trial=0, fname='Results/BiteRateAnalysis/AllInfected')
print("Results saved")