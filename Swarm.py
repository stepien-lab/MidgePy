import numpy as np
import csv
from numba import jit, njit

""" This is the main swarm class, where all midges are simulated. This class holds all attributes of the midges and will
be responsible for moving the host during its move function as well. Moving the time is done by calling the move() method
so be sure not to move the host on their own! This feature will be added later.
"""


class MidgeSwarm:

    def __init__(self, envir, hostswarm, size=100, infected='random', midgedeath=True, eip=21,
                 savepositions=False, tempfile=None):

        self.step = 0  # Initialize the step counter
        self.size = size  # Define the population size of the swarm object
        self.pos_history = []  # Begin a list that tracks the history of midge positions
        self.activeflightvelocity = 0.50  # (m/s) Define the average active velocity of a midge per second
        self.roamflightvelocity = 0.13  # (m/s) Define the average roaming velocity of a midge per second
        self.detectiondistance = 300  # (m) Define the distance at which the midges can detect the host
        self.bitethresholddistance = self.activeflightvelocity  # (m) Define the distance at which a midge must be in order to bite the host
        self.eipconstants = (-0.0636, 0.0069)  # EIP constants for linreg line fit on BTV10 in Wittman et. al.
        self.dpsconstants = (-0.0681, 0.0064)  # DPS constants for linreg line fit on temperature with 85% humidity in Wittman et. al.
        self.midgebitesperstep = []  # Keep track of the midge bites each time step
        self.totalinfectedmidges = []  # Keep track of the total number of infected midges
        self.infecteddeaths = []  # Keep track of the number of infected midges that die each step (only if midgedeath is True)
        self.uninfecteddeaths = []  # Keep track of the number of uninfected midges that die each step (only if midgedeath is True)
        self.infectedbitesperstep = []  # Track the total number of bites from infected midges per step
        self.btvincubating = np.full(self.size,
                                     False)  # Define the array that tracks whether BTV is incubating inside the midge
        self.incubationstarttime = np.full(self.size,
                                           0)  # Create an array that tracks when midges begin incubation for BTV
        self.envir = envir  # Attach the environment object to the swarm class
        self.hostswarm = hostswarm  # The midge swarm class will take the host swarm to know the locations and attributes of each host
        self.daylength = 300  # The length in minutes of a single day (note it is not the entire day, only the length of each period simulated
        self.biterate = 2 * self.daylength  # This variable determines how often a midge is expected to bite a host
        self.timeoffeeding = np.random.randint(-self.biterate, 0,
                                               self.size)  # List to keep track of the time when each midge has fed
        self.pVtoH = 0.90  # Probability of transmission of BTV from a vector to the host
        self.pHtoV = 0.14  # Probability of transmission of BTV from a host to the vector
        self.savepositions = savepositions  # Save each midge's position history throughtout the simulation (MUST BE TRUE IF SAVING MIDGE POSITIONS)

        self.midgedeath = midgedeath  # Enable this if you would like to simulate midges dying and being replaced by new ones

        # Create a random positions array for the midges if desired, otherwise it is defined
        self.positions = np.random.uniform(low=0.0, high=envir.length, size=(self.size, 2))

        self.randomvector = generate_random_vector(self.envir.length, self.size,
                                                   self.positions)  # Array of random vector where the midges travel, updates every few steps

        # Create a random array of which midges are infected if desired, otherwise it is defined
        if infected == 'random':
            self.infected = np.random.choice([True, False], self.size, p=[0.01, 0.99])
        else:
            self.infected = infected

        # If a temperature file has been linked (in Celsius), add this to the temperature variable
        self.temperature = None
        if tempfile != None:
            self.temperature = np.genfromtxt(tempfile, delimiter=",")
        else:
            # Otherwise set the temperature to 25 degrees Celsius
            self.temperature = np.full(100, 25)

        # Compute EIP as given in Wittman et. al. with the regression line
        self.eip = 1 / (self.eipconstants[0] + self.eipconstants[1] * self.temperature)

        # Compute daily survival rate as given in Wittman et. al. with the provided hazard function
        self.dps = 1 - (self.dpsconstants[0] + self.dpsconstants[1] * self.temperature)

    # The step function that calculates all movement (dt is given in seconds)
    def move(self, dt=1):

        if self.savepositions:
            self.totalinfectedmidges.append(np.sum(self.infected))

        # Update the infected midges to be those that have completed their EIP
        self.infected = np.logical_or(self.infected, np.logical_and(self.incubationstarttime != 0, np.abs(
            self.incubationstarttime - self.step) >= self.daylength * self.eip[int(self.step // self.daylength)]))

        # Do the same for the hosts
        self.hostswarm.infected = np.logical_or(self.hostswarm.infected, np.logical_and(
            self.hostswarm.incubationstarttime != 0,
            np.abs(self.hostswarm.incubationstarttime - self.step) >= (self.daylength * self.hostswarm.incubationtime)))

        # Move the host once every day
        if self.step % self.daylength == 0:
            self.hostswarm.move()

            if self.step == 0:
                self.infecteddeaths.append(0)
                self.uninfecteddeaths.append(0)

            # Replace some midges once per day if self.midgedeath is enabled
            if self.midgedeath:
                survivingmidges = np.random.choice([True, False], self.size,
                                                   p=[self.dps[int(self.step // self.daylength)],
                                                      1 - self.dps[int(self.step // self.daylength)]])
                newpositions = np.random.uniform(low=0.0, high=self.envir.length, size=(self.size, 2))

                self.infecteddeaths.append(np.sum(~survivingmidges * self.infected))
                self.uninfecteddeaths.append(np.sum(~survivingmidges * ~self.infected))

                # Give the midges new positions and reset all other parameters
                self.btvincubating *= survivingmidges
                self.infected *= survivingmidges
                self.incubationstarttime *= survivingmidges
                survivingmidges = np.expand_dims(survivingmidges, 1)
                self.positions = self.positions * survivingmidges + newpositions * (~survivingmidges)

        # A new random vector is generated every 30 minutes for the midges to travel in
        if self.step % 30 == 0:
            self.randomvector = generate_random_vector(self.envir.length, self.size, self.positions)

        # Calculate which midges have fed lately by tracking when the last bloodmeal was
        self.fed = ((self.step - self.timeoffeeding) < self.biterate)

        # Find the matrix of vectors from each midge to each host (self.size x self.hostswarm.size matrix)
        targetmatrix = self.calculate_target_matrix()

        # TODO: REPLACE WITH ACCELERATED FUNCTION

        # Calculate the matrix of distances from each midge to each host and find the closest host
        distancematrix = np.sqrt(np.square(targetmatrix[:, 0, :, 0]) + np.square(targetmatrix[:, 1, :, 1]))

        closesthost = np.argmin(distancematrix, axis=1)

        # Find the directions for each midge by finding vector from the closest host
        midgedirections = np.empty(shape=(self.size, 2), dtype=float)

        for i in range(self.size):
            midgedirections[i] = (targetmatrix[i, 0, closesthost[i], 0], targetmatrix[i, 1, closesthost[i], 1])

        # Calculate the distance that each midge must travel to reach the closest host
        hostdistances = np.linalg.norm(midgedirections, axis=1)

        # The list of midges that are within the detection distance of their closest host and have not fed
        detectinghost = (hostdistances < self.detectiondistance) * ~self.fed

        # Calculate the new positions by using the flightvelocity variable
        self.positions = self.get_positions() + self.activeflightvelocity * dt * ((np.expand_dims(detectinghost, 1) *
                                                                                   np.divide(midgedirections,
                                                                                             np.expand_dims(
                                                                                                 hostdistances, 1),
                                                                                             out=np.zeros_like(
                                                                                                 midgedirections),
                                                                                             where=np.expand_dims(
                                                                                                 hostdistances,
                                                                                                 1) != 0)) + (
                                                                                          self.randomvector * ~np.expand_dims(
                                                                                      detectinghost, 1)))

        # Calculate which midges will feed and the results of their feeding
        self.feed(closesthost, hostdistances, dt)

        # Append the position history to pos_history
        if self.savepositions:
            self.pos_history.append(self.get_positions())
            self.hostswarm.pos_history.append(self.hostswarm.get_positions())

        # Increment the step counter
        self.step += 1

    # Returns the matrix of vectors from every midge to every host (midges x host x 2) size
    def calculate_target_matrix(self):
        pos = self.hostswarm.get_positions()

        answer = -np.subtract.outer(self.get_positions(), pos)

        return answer

    # Returns the numpy array of positions
    def get_positions(self):
        return self.positions

    # Returns the numpy array of infected midges
    def get_infected(self):
        return self.infected

    # Returns the HostSwarm object
    def get_hostswarm(self):
        return self.hostswarm

    # Returns the full position history of the midges
    def get_full_pos_history(self):
        return [*self.pos_history, self.get_positions()]

    def feed(self, closesthost, hostdistances, dt):

        # Find which midges are close enough to the host to bite them and they have not recently fed
        feedingmidges = (hostdistances[closesthost] < self.bitethresholddistance * dt) & ~self.fed

        if self.savepositions:
            self.infectedbitesperstep.append(np.sum(feedingmidges * self.infected))

        # TODO: Add consideration for midges already infected
        # The midges will begin BTV incubation if they are feeding and the closest host is infected, do the same for the host
        newincubation = np.random.choice([True, False], self.size, p=[self.pHtoV, 1 - self.pHtoV]) * (
                feedingmidges & self.hostswarm.infected[closesthost])

        # Start incubation in the midges if they have not already begun
        self.incubationstarttime[newincubation & ~self.btvincubating] = self.step

        # Add the newly incubating midges to the list of btvincubating midges
        self.btvincubating = np.logical_or(self.btvincubating, newincubation)

        # Create the probability of infection array that determines which host will become infected if bitten during this timestep
        hostinfectedprob = np.random.rand(self.size) < self.pVtoH
        # Track which host become inoculated (if they are bitten, midge is infected, probability is favorable, and have not already been inoculated)
        # for i in range(self.size):
        #     if feedingmidges[i] & self.infected[i] & infectedprob[i] & (self.hostswarm.incubationstarttime[closesthost[i]] == 0):
        #         # The host can be infected with probability infectedprob from a single bite from an infected midge
        #         self.hostswarm.incubationstarttime[closesthost[i]] = self.step

        self.hostswarm.incubationstarttime = determineincubation(self.step, self.size, feedingmidges, self.infected,
                                                                 hostinfectedprob, self.hostswarm.incubationstarttime,
                                                                 closesthost)

        # Update time of feeding to the current step for the midges which have just fed
        self.timeoffeeding[feedingmidges] = self.step

        # Append the midge bites for this time step and total infected midges
        if self.savepositions:
            self.midgebitesperstep.append(feedingmidges.sum())

        self.hostswarm.totalinfectedhost.append(self.hostswarm.infected.sum())

    def writetocsv(self, trial=None, fname='Results/midgesim'):

        fname = fname + 'DPS' + str(int(100*self.dps[int(self.step // self.daylength)])) + 'Trial' + str(trial) + '.csv'

        with open(fname, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(
                ['Step', 'Day', 'Infected Midges', 'Infected Midges %', 'Infected Host', 'Infected Host %',
                 'Midge Bites', 'Infected Midge Bites',
                 'VF', 'VR', 'DD', 'EIP', 'PVTH', 'PHTV', 'DPS', 'PD', 'BR',
                 'MDR', 'IIM', 'Infected Deaths', 'Uninfected Deaths'])
            for i in range(self.step):
                if self.midgedeath:
                    writer.writerow([i, i / float(self.daylength), self.totalinfectedmidges[i],
                                     self.totalinfectedmidges[i] / self.size * 100,
                                     self.hostswarm.totalinfectedhost[i],
                                     self.hostswarm.totalinfectedhost[i] / self.hostswarm.size * 100,
                                     self.midgebitesperstep[i], self.infectedbitesperstep[i], self.activeflightvelocity,
                                     self.roamflightvelocity, self.detectiondistance, self.eip[self.step // self.daylength], self.pVtoH,
                                     self.pHtoV, self.dps[int(self.step // self.daylength)], self.hostswarm.size, self.biterate,
                                     self.size // self.hostswarm.size,
                                     self.totalinfectedmidges[0], self.infecteddeaths[i // self.daylength],
                                     self.uninfecteddeaths[i // self.daylength]])

                else:
                    writer.writerow([i, i / float(self.daylength), self.totalinfectedmidges[i],
                                     self.totalinfectedmidges[i] / self.size * 100,
                                     self.hostswarm.totalinfectedhost[i],
                                     self.hostswarm.totalinfectedhost[i] / self.hostswarm.size * 100,
                                     self.midgebitesperstep[i], self.activeflightvelocity, self.roamflightvelocity,
                                     self.detectiondistance, self.eip[self.step // self.daylength], self.pVtoH,
                                     self.pHtoV, self.dps[int(self.step // self.daylength)], self.hostswarm.size, self.biterate,
                                     self.size // self.hostswarm.size,
                                     self.totalinfectedmidges[0]])
        return

    def SavePositions(self, fnamemidge, fnamehost):
        # SAVE MIDGE POSITIONS
        with open(fnamemidge, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            header = ['Step', 'Midge', 'Midge X', 'Midge Y']
            writer.writerow(header)
            for i in range(self.step):
                for j in range(self.size):
                    writer.writerow([i, j, self.pos_history[i][j][0], self.pos_history[i][j][1]])

        # SAVE DEER POSITIONS
        with open(fnamehost, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            header = ['Step', 'Host', 'Host X', 'Host Y']
            writer.writerow(header)
            for i in range(self.step):
                for j in range(self.hostswarm.size):
                    writer.writerow([i, j, self.hostswarm.pos_history[i][j][0], self.hostswarm.pos_history[i][j][1]])

        return


class HostSwarm:

    def __init__(self, envir, size=50, positions='random', infected='random', steplength=1.0):

        # Define the population size of the swarm object
        self.size = size

        # Define the average step length of a host per time step
        self.avgsteplength = steplength

        # Keep track of the total number of infected host
        self.totalinfectedhost = []

        # Attach the environment object to the swarm class
        self.envir = envir

        self.incubationtime = 2

        self.incubationstarttime = np.full(self.size,
                                           0)  # Create an array that tracks when midges begin incubation for BTV

        self.pos_history = []

        # Create a random positions array for the host if desired, otherwise it is defined
        if positions == 'random':
            self.positions = np.random.uniform(low=0.0, high=envir.length, size=(self.size, 2))
        else:
            self.positions = positions

        # Create a random array of which host begin infected if desired, otherwise it is defined
        if infected == 'random':
            self.infected = np.full(self.size, False)
            # self.infected = np.random.choice([True, False], self.size, p=[0.2, 0.8])
        else:
            self.infected = infected

    # Move function that is called by the MidgeSwarm class, generates a new set of points for the host (random)
    def move(self):
        self.positions = np.random.uniform(low=0.0, high=self.envir.length, size=(self.size, 2))

    # Returns the numpy array of positions
    def get_positions(self):
        return self.positions


# Random movement function
def generate_random_vector(length, size, positions):
    # Creates a vector from the midge to a random position within the domain, then the midge will follow that vector
    newvectors = np.random.uniform(low=0.0, high=length, size=(size, 2)) - positions
    newvectors /= np.expand_dims(np.linalg.norm(newvectors, axis=1), axis=1)

    return newvectors

@jit
def determineincubation(step, length, feedingmidges, infected, infectedprob, hostswarmincubationstarttime, closesthost):
    qualified = feedingmidges & infected & infectedprob
    for i in range(length):
        if qualified[i] & (hostswarmincubationstarttime[closesthost[i]] == 0):
            # The host can be infected with probability infectedprob from a single bite from an infected midge
            hostswarmincubationstarttime[closesthost[i]] = step

    return hostswarmincubationstarttime
