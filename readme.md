# Putting energy back in control with CBFs 
Using a port-Hamiltonian formalism, we show the qualitative and quantitative effect of safety-critical control implemented with control barrier functions (CBFs) on the power balance of controlled physical systems. The presented results will provide novel tools to design CBFs inducing desired energetic behaviors of the closed-loop system, including nontrivial damping injection effects and non-passive control actions, effectively injecting energy in the system in a controlled manner. Simulations validate the stated results.

## Installation ##

Setup virtual environment

```
conda env create -f environment.yml
```

Install Package

```
pip install -e .
```


## Notes
If the trajectory pass from p=0 (i.e. zero velocity) while it is out of the safe set, then the optimization fails.


## TODO
[] probably not working for non-scalar systems (e.g. q.shape = (2,1))
 