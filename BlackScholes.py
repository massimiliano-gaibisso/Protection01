import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

class BlackScholesCalculator:
    def __init__(self, S, K, T, r, sigma, q=0):
        """
        Initialize Black-Scholes calculator
        
        Parameters:
        S: float - Spot price
        K: float - Strike price
        T: float - Time to maturity (in years)
        r: float - Risk-free interest rate (annual)
        sigma: float - Volatility (annual)
        q: float - Dividend yield (annual)
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        
        
        # Calculate d1 and d2
        self._update_d1_d2()
    
    def _update_d1_d2(self):
        """Update d1 and d2 values used in calculations"""
        self.d1 = (np.log(self.S/self.K) + (self.r - self.q + 0.5*self.sigma**2)*self.T) / (self.sigma*np.sqrt(self.T))
        self.d2 = self.d1 - self.sigma*np.sqrt(self.T)
    
    def call_price(self):
        """Calculate call option price"""
        return (self.S * np.exp(-self.q*self.T) * norm.cdf(self.d1) - 
                self.K * np.exp(-self.r*self.T) * norm.cdf(self.d2))
    
    def put_price(self):
        """Calculate put option price"""
        return (self.K * np.exp(-self.r*self.T) * norm.cdf(-self.d2) - 
                self.S * np.exp(-self.q*self.T) * norm.cdf(-self.d1))
    
    # ITM probability
    def itm_prob(self, option_type='call'):        
        """Calculate ITM probability"""
        if option_type.lower() == 'call':
            return norm.cdf(self.d2)
        else:
            return norm.cdf(-self.d2)
    
    # Greeks calculations
    def delta(self, option_type='call'):
        """Calculate Delta"""
        if option_type.lower() == 'call':
            return np.exp(-self.q*self.T) * norm.cdf(self.d1)
        else:
            return np.exp(-self.q*self.T) * (norm.cdf(self.d1) - 1)
    
    def gamma(self):
        """Calculate Gamma (same for call and put)"""
        return (np.exp(-self.q*self.T) * norm.pdf(self.d1)) / (self.S * self.sigma * np.sqrt(self.T))
    
    def theta(self, option_type='call'):
        """Calculate Theta"""
        term1 = -(self.S * self.sigma * np.exp(-self.q*self.T) * norm.pdf(self.d1)) / (2 * np.sqrt(self.T))
        if option_type.lower() == 'call':
            term2 = -self.r * self.K * np.exp(-self.r*self.T) * norm.cdf(self.d2)
            term3 = self.q * self.S * np.exp(-self.q*self.T) * norm.cdf(self.d1)
        else:
            term2 = self.r * self.K * np.exp(-self.r*self.T) * norm.cdf(-self.d2)
            term3 = -self.q * self.S * np.exp(-self.q*self.T) * norm.cdf(-self.d1)
        return term1 + term2 + term3
    
    def vega(self):
        """Calculate Vega (same for call and put)"""
        return self.S * np.exp(-self.q*self.T) * np.sqrt(self.T) * norm.pdf(self.d1)
    
    def rho(self, option_type='call'):
        """Calculate Rho"""
        if option_type.lower() == 'call':
            return self.K * self.T * np.exp(-self.r*self.T) * norm.cdf(self.d2)
        else:
            return -self.K * self.T * np.exp(-self.r*self.T) * norm.cdf(-self.d2)

    def plot_sensitivity(self, greek_func, param_range, param_name, option_type='call'):
        """
        Plot option sensitivity to a parameter
        
        Parameters:
        greek_func: function - The Greek to plot
        param_range: array - Range of parameter values to plot
        param_name: str - Name of parameter being varied ('S', 'sigma', 'r', 'q')
        option_type: str - 'call' or 'put'
        """
        original_value = getattr(self, param_name)
        greek_values = []
        
        for value in param_range:
            setattr(self, param_name, value)
            self._update_d1_d2()
            if greek_func.__name__ in ['gamma', 'vega']:
                greek_values.append(greek_func())
            else:
                greek_values.append(greek_func(option_type))
        
        # Reset to original value
        setattr(self, param_name, original_value)
        self._update_d1_d2()
        
        plt.figure(figsize=(10, 6))
        plt.plot(param_range, greek_values)
        plt.title(f'{greek_func.__name__.capitalize()} vs {param_name}')
        plt.xlabel(param_name)
        plt.ylabel(greek_func.__name__.capitalize())
        plt.grid(True)
        plt.show()

def example_usage():
    # Example parameters
    S = 100  # Spot price
    K = 100  # Strike price
    T = 1    # Time to maturity (1 year)
    r = 0.05 # Risk-free rate (5%)
    sigma = 0.2  # Volatility (20%)
    q = 0.02 # Dividend yield (2%)
    
    # Create calculator instance
    bs = BlackScholesCalculator(S, K, T, r, sigma, q)
    
    # Calculate option prices
    print(f"Call Price: {bs.call_price():.4f}")
    print(f"Put Price: {bs.put_price():.4f}")
    
    # Plot Delta sensitivity to spot price
    spot_range = np.linspace(80, 120, 100)
    bs.plot_sensitivity(bs.delta, spot_range, 'S', 'call')
    
    # Plot Vega sensitivity to volatility
    vol_range = np.linspace(0.1, 0.4, 100)
    bs.plot_sensitivity(bs.vega, vol_range, 'sigma')