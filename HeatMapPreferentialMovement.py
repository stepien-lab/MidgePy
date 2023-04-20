import numpy as np
import SwarmPreferentialMovement as Swarm
import Environment
import threading
import multiprocessing
import time


def StorePositions(days):
    # np.random.seed()
    midgehostratio = 500
    hostpop = 100
    midgepop = hostpop * midgehostratio
    midges = np.full(midgepop, False)
    midges[0:5] = True
    hostinf = np.full(hostpop, False)
    print('Creating Environment')
    envir = Environment.Envir(length=1000)
    print('Creating Host Swarm')
    host = Swarm.HostSwarm(envir=envir, size=hostpop, infected=hostinf)
    print('Creating Midge Swarm')
    swrm = Swarm.MidgeSwarmPreferentialMovement(envir=envir, size=midgepop, mapimage='FarmMap.png', hostswarm=host, infected=midges, savepositions=True, dps=0.75, eip=15)
    dt = 60  # Step the simulation every 60 seconds (1 minute)
    daylength = 300  # Number of steps in each day
    steps = daylength * days  # Total number of steps for the simulation
    for k in range(steps):
        swrm.move(dt)
        if k % 300 == 0:
            print('Day', k % 300, 'completed')
            # time.sleep(0.1)
        print("Step", k)

    # After the set number of steps, save all midge positions to a file
    swrm.SavePositions(fnamemidge='/blue/rcstudents/shanegladson/HeatMap/MidgePositionsPreferentialMovement.csv', fnamehost='/blue/rcstudents/shanegladson/HeatMap/HostPositionsPreferentialMovement.csv')

if __name__ == '__main__':
    print('Preferential Movement Begun')
    time.sleep(0.5)
    StorePositions(30)
