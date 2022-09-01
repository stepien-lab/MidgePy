import numpy as np


# Environment class that may have terrain attributes in the future
class Envir():
    # TODO: AT SOME POINT ADD SUPPORT FOR WIND IF DESIRED
    def __init__(self, length=1000):
        # Set the size of the environment (a square for now)
        self.length = length
