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
swrm = Swarm.MidgeSwarm(envir=envir, size=midgepop, hostswarm=host, infected=midges, savepositions=True,
                        tempfile='temperature.csv')
swrm.pVtoH = 0  # Don't want to consider transmission to host
dt = 60  # Step the simulation every 60 seconds (1 minute)
days = 60  # Total length in days of the simulation
length = days * 300  # Total length in time steps of the simulation

print("Moving swarm...")
# RUN UNTIL ALL INFECTED MIDGES (FIRST GEN) HAVE DIED
for i in range(length):
    swrm.move(dt)

    if i % 300 == 0:
        print("Day", i // 300)

print("Simulation finished")

print("Saving results...")
swrm.SavePositions(fnamemidge='Results/Midge Paths/midgepositions.csv', fnamehost='Results/Midge Paths/hostpositions.csv')
print("Results saved")