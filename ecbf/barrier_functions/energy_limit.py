 
from ecbf.scripts.basecbf import BaseCBF

class EnergyLimit(BaseCBF):
    def __init__(self, energy_func, c, num_states=2, pump=False): 
        '''
        This class limits the energy of the system to a certain value "c". 
        If "pump" is set to True, "c" is the minimum energy of the system. Otherwise, "c" is the maximum energy of the system.
        '''
        super().__init__()
        self.energy_func = energy_func  

        self.c = c

        self.num_states = num_states

        self.pump = pump

    def function(self, state): 
        q = state[0] if self.num_states==2 else state[0:self.num_states//2]
        p = state[1] if self.num_states==2 else state[self.num_states//2:]
        if self.pump:
            _h = self.energy_func(q, p) - self.c
        else: 
            _h = self.c - self.energy_func(q, p) 
        return _h 
     