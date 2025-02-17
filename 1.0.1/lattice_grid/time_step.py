

class Time_Step:
    def __init__(self, step : int, ta : float, tb : float):
        
        self.step = step #number of steps
        self.ta = ta #Time Starts
        self.tb = tb #Time Ends

    def time(self):
        return (self.tb - self.ta)/self.step