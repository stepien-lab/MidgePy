# MidgePy

<a href="https://github.com/stepien-lab/MidgePy/"><img src="https://img.shields.io/badge/GitHub-MidgePy-blue" /></a> <a href="https://doi.org/10.1101/2022.09.26.509502"><img src="https://img.shields.io/badge/bioRxiv-2022.09.26.509502-orange" /></a> <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>

The code contained in the MidgePy project was developed to numerically simulate an agent-based model of biting midge dynamics to understand Bluetongue outbreaks. It is described in:
>[Shane Gladson](https://github.com/shanegladson) and [Tracy L. Stepien](https://github.com/tstepien/), An agent-based model of biting midge dynamics to understand Bluetongue outbreaks, submitted to _Bulletin of Mathematical Biology_, bioRxiv: [2022.09.26.509502](https://doi.org/10.1101/2022.09.26.509502).

## Programs
The Python libraries required to run the model include NumPy, SciPy, Matplotlib, Seaborn, SALib, and Numba.
+ [Main.py](Main.py): run this script to perform sensitivity analysis on the model using the SALib library. This script will produce a large number of processes to simulate the model and will save the data to <i>csv</i> format. Change the desired number of processes if too many are created.
+ [Outbreak.py](Outbreak.py): run this script to find the probability of outbreak given different values for $\alpha$ and $I_0$ (reference the publication for clarification on the parameters). This script will produce a large number of processes to simulate the model and will save the data to <i>csv</i> format. Change the desired number of processes if too many are created.
+ [HeatMap.py](HeatMap.py): run this script to generate the data for the heatmap used in the publication. This script runs multiple 60 day simulations at desired levels of $\alpha$ and $\rho$. This script will produce a large number of processes to simulate the model and will save the data to <i>csv</i> format. Change the desired number of processes if too many are created.
+ [TrackMidges.py](TrackMidges.py): Run this script to generate a sample of the flight paths of all midges over 2 days. This output was used to generate the sample midge flight path in the publication. This script will save the data to <i>csv</i> format.

## Lead Developer
The lead developer of this code is [Shane Gladson](https://github.com/shanegladson).

## Licensing
Copyright 2022 [Tracy Stepien's Research Lab Group](https://github.com/stepien-lab/). This is free software made available under the MIT License. For details see the [LICENSE](LICENSE) file.
