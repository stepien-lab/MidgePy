import numpy as np
import seaborn as sns
import Swarm
import Environment
from SALib.sample import saltelli
import multiprocessing
from multiprocessing.pool import Pool

sns.set_style('whitegrid')


def SimMidges(iim, dps, eip):
    midgehostratio = 100  # Midge/host ratio

    hostpop = 100
    midgepop = hostpop * midgehostratio

    midges = np.full(midgepop, False)
    midges[0:iim] = True  # Let some midges be infected with BTV

    hostinf = np.full(hostpop, False)  # Entire host population is naive to BTV

    envir = Environment.Envir(length=1000)
    host = Swarm.HostSwarm(envir=envir, size=hostpop, infected=hostinf)
    swrm = Swarm.MidgeSwarm(envir=envir, size=midgepop, hostswarm=host, infected=midges, dps=dps, eip=eip)
    dt = 60  # Step the simulation every 60 seconds (1 minute)
    steps = 300 * 1  # Total number of steps for the simulation

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
    'num_vars': 2,
    'names': ['dps', 'eip'],
    'bounds': [
        [0.6, 0.9],
        [10, 20]
    ]
}

params = saltelli.sample(problem, 16)


def SaveAnalysis(iim, i):
    results = np.empty(shape=(params.shape[0], params.shape[1] + 1))
    inputs = []
    for j, X in enumerate(params):
        dps, eip = X
        inputs.append((iim, dps, eip))
    with Pool() as pool:
        for result in pool.imap(SimMidges, inputs):
            results.append(result)
    np.savetxt(fname='Results/IIM' + str(iim) + '/Trial' + str(i) + '.csv', X=results, delimiter=',', newline='\n')


threadlist = []

if __name__ == '__main__':
    for iim in range(1, 2):
        for i in range(1):
            p = multiprocessing.Process(target=SaveAnalysis, args=(iim, i))
            p.start()
            threadlist.append(p)

        for p in threadlist:
            p.join()
