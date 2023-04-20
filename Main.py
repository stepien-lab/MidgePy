import numpy as np
import seaborn as sns
import Swarm
import Environment
from SALib.sample import saltelli
import multiprocessing
from multiprocessing.pool import Pool
import numpy as np
import time
import os

sns.set_style('whitegrid')


def SimMidges(iim, dps, eip, pVtoH, pHtoV, incubationtime):
    np.random.seed()  # Add a random seed to prevent result duplication
    midgehostratio = 100  # Midge/host ratio

    hostpop = 100
    midgepop = hostpop * midgehostratio

    midges = np.full(midgepop, False)
    midges[0:iim] = True  # Let some midges be infected with BTV

    hostinf = np.full(hostpop, False)  # Entire host population is naive to BTV

    envir = Environment.Envir(length=1000)
    host = Swarm.HostSwarm(envir=envir, size=hostpop, infected=hostinf, incubationtime=incubationtime)
    swrm = Swarm.MidgeSwarm(envir=envir, size=midgepop, hostswarm=host, infected=midges, dps=dps, eip=eip,
                            pVtoH=pVtoH, pHtoV=pHtoV)
    dt = 60  # Step the simulation every 60 seconds (1 minute)
    steps = 300 * 60  # Total number of steps for the simulation

    print("Moving swarm...")
    for i in range(steps):
        swrm.move(dt)

        if i % 300 == 0:
            print("Day", i // 300)

    print("Simulation finished")

    # print("Saving results...")
    # swrm.writetocsv(trial=j)
    # print("Results saved")

    return swrm.hostswarm.totalinfectedhost[-1]


problem = {
    'num_vars': 5,
    'names': ['dps', 'eip', 'pVtoH', 'pHtoV', 'incubationtime'],
    'bounds': [
        [0.0, 1.0],
        [0, 20],
        [0.0, 1.0],
        [0.0, 1.0],
        [1.0, 10.0]
    ]
}

params = saltelli.sample(problem, 16)
print(params.shape)


def SaveAnalysis(iim, i):
    results = []
    inputs = []
    for j, X in enumerate(params):
        dps, eip, pVtoH, pHtoV, incubationtime = X
        inputs.append((iim, dps, eip, pVtoH, pHtoV, incubationtime))
    with Pool() as pool:
        print('CPU count:', os.cpu_count())
        for result in pool.starmap(SimMidges, inputs):
            results.append(result)

    np.savetxt(fname='/blue/rcstudents/shanegladson/IIM' + str(iim) + '/Trial' + str(i) + '.csv', X=results, delimiter=',', newline='\n')


threadlist = []

if __name__ == '__main__':
    for iim in range(1, 6):
        for i in range(15):
            # p = multiprocessing.Process(target=SaveAnalysis, args=(iim, i))
            # p.start()
            # threadlist.append(p)
            SaveAnalysis(iim, i)

        # for p in threadlist:
        #     p.join()
