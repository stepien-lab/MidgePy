import numpy as np
import Swarm
import Environment
import threading
import multiprocessing
import time


def Outbreak(dps, eip, iim):
    midgehostratio = 100  # Midge/host ratio

    hostpop = 100
    midgepop = hostpop * midgehostratio

    midges = np.full(midgepop, False)
    midges[0:iim] = True  # Let some midges be infected with BTV

    hostinfected = np.full(hostpop, False)  # Entire deer population is naive to BTV

    envir = Environment.Envir(length=1000)
    host = Swarm.HostSwarm(envir=envir, size=hostpop, infected=hostinfected)
    swrm = Swarm.MidgeSwarm(envir=envir, size=midgepop, hostswarm=host, infected=midges, dps=dps, eip=eip)
    dt = 60  # Step the simulation every 60 seconds (1 minute)

    while True:
        if np.sum(swrm.get_infected()) == 0 and (np.sum(swrm.hostswarm.incubationstarttime) == 0):
            return 0
        elif np.sum(swrm.hostswarm.incubationstarttime) != 0:
            return 1

        swrm.move(dt)


numsims = 100
iim = [1, 2, 3, 4, 5, 15, 50, 100]

dps = np.linspace(0.0, 1.0, num=51, endpoint=True)
print(dps)
eip = 14  # NOT IMPORTANT FOR THIS MODEL
numoutbreaks = np.empty(shape=(dps.shape[0], 2), dtype=float)


def ThreadSet(iim):
    for i in range(dps.shape[0]):
        success = np.empty(shape=numsims, dtype=float)
        for j in range(numsims):
            success[j] = Outbreak(dps=dps[i], eip=eip, iim=iim)
            if j % (numsims + 1) / 5 == 0:
                print('IIM', iim, 'DPS:', dps[i] + 1, 'Simulation:', j)

        numoutbreaks[i, 0] = dps[i]
        numoutbreaks[i, 1] = np.sum(success)

    print(numoutbreaks)
    np.savetxt('Results/IIM' + str(iim) + '/OutbreakProbabilityLonger.csv', X=numoutbreaks, delimiter=',', newline='\n')


threadlist = []

if __name__ == '__main__':
    for i in iim:
        p = multiprocessing.Process(target=ThreadSet, args=(i,))
        p.start()
        threadlist.append(p)

    for p in threadlist:
        p.join()
