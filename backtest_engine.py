import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class OptimalExecutionStrategy:
    def __init__(self, lob_data_path):
        """
        Initialize the optimal execution strategy with LOB data
        
        Args:
            lob_data_path (str): Path to the limit order book data file
        """
        # Load and preprocess LOB data
        self.df = pd.read_csv(lob_data_path)
        self.preprocess_data()
        
        # Model parameters (calibrated from Problem 1)
        self.alpha = 0.2243   # Spread coefficient
        self.beta = 0.0565    # Depth coefficient
        self.gamma = 0.2184   # Power-law exponent
        
        # Execution parameters
        self.max_depth_utilization = 0.2  # Max % of depth to consume per period
        self.volatility_threshold = 2.0    # Multiplier of spread for volatility check
        
    def preprocess_data(self):
        """Preprocess the LOB data"""
        # Calculate mid price and spread
        self.df['mid_price'] = (self.df['bid_px_00'] + self.df['ask_px_00']) / 2
        self.df['spread'] = self.df['ask_px_00'] - self.df['bid_px_00']
        
        # Calculate depth (minimum of bid and ask depth at best levels)
        self.df['depth'] = self.df[['bid_sz_00', 'ask_sz_00']].min(axis=1)
        
        # Calculate price returns for volatility estimation
        self.df['returns'] = self.df['mid_price'].pct_change()
        
    def temporary_impact(self, x, spread, depth):
        """
        Calculate temporary impact cost for executing x shares
        
        Args:
            x (float): Number of shares to execute
            spread (float): Current bid-ask spread
            depth (float): Current depth at best bid/ask
            
        Returns:
            float: Temporary impact cost per share
        """
        return (self.alpha * spread + self.beta / np.sqrt(depth)) * np.power(x, self.gamma)
    
    def total_execution_cost(self, x, mid_price, spread, depth):
        """
        Calculate total execution cost for x shares
        
        Args:
            x (float): Number of shares to execute
            mid_price (float): Current mid price
            spread (float): Current bid-ask spread
            depth (float): Current depth at best bid/ask
            
        Returns:
            float: Total execution cost
        """
        impact = self.temporary_impact(x, spread, depth)
        return x * (mid_price + impact)
    
    def optimize_period_execution(self, remaining_shares, periods_left, current_lob_state):
        """
        Optimize execution for current period
        
        Args:
            remaining_shares (float): Shares remaining to execute
            periods_left (int): Number of periods remaining
            current_lob_state (pd.Series): Current LOB state
            
        Returns:
            float: Optimal shares to execute this period
        """
        # Extract current market conditions
        spread = current_lob_state['spread']
        depth = current_lob_state['depth']
        mid_price = current_lob_state['mid_price']
        
        # Define optimization problem
        def cost(x):
            return self.total_execution_cost(x[0], mid_price, spread, depth)
        
        # Constraints: must execute all shares by end
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - remaining_shares}
        )
        
        # Bounds: can't execute negative or more than remaining shares
        bounds = [(0, remaining_shares)]
        
        # Initial guess: equal distribution over remaining periods
        x0 = np.array([remaining_shares / periods_left])
        
        # Solve optimization
        res = minimize(
            cost,
            x0=x0,
            bounds=bounds,
            constraints=constraints,
            method='SLSQP'
        )
        
        return res.x[0]
    
    def apply_market_constraints(self, x_optimal, current_lob_state, remaining_shares, periods_left):
        """
        Apply market constraints to the optimal execution quantity
        
        Args:
            x_optimal (float): Optimally calculated execution quantity
            current_lob_state (pd.Series): Current LOB state
            remaining_shares (float): Shares remaining to execute
            periods_left (int): Periods remaining in execution
            
        Returns:
            float: Constrained execution quantity
        """
        # Don't take more than allowed percentage of current depth
        max_by_depth = self.max_depth_utilization * current_lob_state['depth']
        x_constrained = min(x_optimal, max_by_depth)
        
        # Fall back to VWAP-like execution if high volatility
        if current_lob_state.name > 0:  # Check if not first period
            last_price = self.df.iloc[current_lob_state.name - 1]['mid_price']
            current_price = current_lob_state['mid_price']
            price_change = abs(current_price - last_price)
            
            if price_change > self.volatility_threshold * current_lob_state['spread']:
                # High volatility - execute evenly over remaining periods
                x_constrained = min(x_constrained, remaining_shares / periods_left)
        
        return x_constrained
    
    def execute(self, total_shares, start_period=0):
        """
        Execute optimal strategy for given number of shares
        
        Args:
            total_shares (float): Total shares to execute
            start_period (int): Starting period index
            
        Returns:
            tuple: (execution_schedule, execution_costs)
        """
        execution_schedule = []
        execution_costs = []
        remaining_shares = total_shares
        
        for t in range(start_period, len(self.df)):
            if remaining_shares <= 0:
                break
                
            current_lob = self.df.iloc[t]
            periods_left = len(self.df) - t
            
            # Optimize execution for this period
            x_optimal = self.optimize_period_execution(
                remaining_shares,
                periods_left,
                current_lob
            )
            
            # Apply market constraints
            x_executed = self.apply_market_constraints(
                x_optimal,
                current_lob,
                remaining_shares,
                periods_left
            )
            
            # Calculate impact cost
            impact_cost = self.temporary_impact(
                x_executed, 
                current_lob['spread'], 
                current_lob['depth']
            )
            total_cost = x_executed * (current_lob['mid_price'] + impact_cost)
            
            # Update tracking variables
            execution_schedule.append(x_executed)
            execution_costs.append(total_cost)
            remaining_shares -= x_executed
            
            # Recalibrate model parameters periodically
            if t % 30 == 0:  # Recalibrate every 30 periods
                self.recalibrate_parameters(t)
        
        return np.array(execution_schedule), np.array(execution_costs)
    
    def recalibrate_parameters(self, current_period):
        """
        Recalibrate model parameters using historical data up to current period
        
        Args:
            current_period (int): Current period index
        """
        # In a real implementation, we would refit the model parameters
        # using the available historical data up to current_period
        # For this example, we'll keep the parameters constant
        pass
    
    def analyze_execution(self, execution_schedule, execution_costs):
        """
        Analyze and visualize execution results
        
        Args:
            execution_schedule (np.array): Array of shares executed per period
            execution_costs (np.array): Array of costs incurred per period
        """
        # Calculate cumulative execution
        cumulative_execution = np.cumsum(execution_schedule)
        avg_price = np.sum(execution_costs) / np.sum(execution_schedule)
        
        # Plot execution schedule
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(execution_schedule, label='Shares per period')
        plt.plot(cumulative_execution, label='Cumulative executed')
        plt.xlabel('Time Period')
        plt.ylabel('Shares')
        plt.title('Execution Schedule')
        plt.legend()
        plt.grid(True)
        
        # Plot price impact
        plt.subplot(1, 2, 2)
        impact_per_share = execution_costs / execution_schedule - self.df.iloc[:len(execution_schedule)]['mid_price'].values
        plt.plot(impact_per_share)
        plt.xlabel('Time Period')
        plt.ylabel('Impact per Share ($)')
        plt.title('Price Impact Over Time')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        print(f"Total shares executed: {cumulative_execution[-1]:,.0f}")
        print(f"Average execution price: ${avg_price:,.4f}")
        print(f"Total execution cost: ${np.sum(execution_costs):,.2f}")

# Example usage
if __name__ == "__main__":
    # Initialize strategy with LOB data
    strategy = OptimalExecutionStrategy(r"your_file_path")
    
    # Execute 100,000 shares
    schedule, costs = strategy.execute(100000) #Number of shares which needs to be excecuted
    
    # Analyze results
    strategy.analyze_execution(schedule, costs)