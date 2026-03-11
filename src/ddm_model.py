"""Simplified Drift Diffusion Model implementation."""
import numpy as np
from scipy import stats
from scipy.optimize import differential_evolution
import pandas as pd

class SimpleDDM:
    def __init__(self, v=0.3, a=1.0, t0=300):
        self.v, self.a, self.t0 = v, a, t0
    
    def simulate_trials(self, n_trials=1000, dt=0.001):
        rts = []
        for _ in range(n_trials):
            x, t = self.a/2, 0
            while 0 < x < self.a and t < 5:
                x += self.v*dt + np.sqrt(dt)*np.random.randn()
                t += dt
            rts.append(t*1000 + self.t0)
        return np.array(rts)
    
    def fit(self, rts, verbose=True):
        def objective(params):
            v, a, t0 = params
            if a <= 0 or t0 < 0 or t0 > np.min(rts): return 1e10
            self.v, self.a, self.t0 = v, a, t0
            sim_rts = self.simulate_trials(n_trials=len(rts))
            ks_stat, _ = stats.ks_2samp(rts, sim_rts)
            return ks_stat
        
        bounds = [(0.01, 1.0), (0.5, 3.0), (50, 500)]
        result = differential_evolution(objective, bounds, maxiter=100, seed=42, workers=1, disp=verbose)
        self.v, self.a, self.t0 = result.x
        
        if verbose:
            print(f"Fitted: v={self.v:.3f}, a={self.a:.3f}, t0={self.t0:.1f}ms, KS={result.fun:.4f}")
        return {'v': self.v, 'a': self.a, 't0': self.t0, 'ks_stat': result.fun, 'success': result.success}
    
    def predict_rt_quantiles(self, quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]):
        sim_rts = self.simulate_trials(n_trials=10000)
        return np.quantile(sim_rts, quantiles)
