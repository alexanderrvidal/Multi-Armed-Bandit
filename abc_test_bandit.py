import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class ABCTestBandit:
    def __init__(self, conversion_rates):
        self.conversion_rates = conversion_rates  # Conversion rates for each variant
        
    def display_variant(self, variant):
        # Simulate user conversion (1) or not (0)
        return 1 if np.random.random() < self.conversion_rates[variant] else 0

class ThompsonABCTest:
    # known to perform best in practice for A/B testing
    # intuition: the number of pulls for a given lever should match its actual probability of being the optimal lever.
    def __init__(self, n_variants):
        self.n_variants = n_variants
        # beta distribution parameters for each variant : Uniform prior (alpha=1, beta=1)
        self.successes = np.ones(n_variants)  # Beta prior (alpha)
        self.failures = np.ones(n_variants)   # Beta prior (beta)
        # Add explicit conversion tracking for reporting
        self.conversions = np.zeros(n_variants)
        self.counts = np.zeros(n_variants)
        
    def choose_variant(self):
        # sample from the posterior distribution of each variant
        sampled_rates = [np.random.beta(self.successes[i], self.failures[i]) 
                        for i in range(self.n_variants)]
        # returns arm with highest sampled value
        return np.argmax(sampled_rates)
    
    def update(self, variant, converted):
        #if those chosen arm is converted, update the alpha parameter
        if converted:
            self.successes[variant] += 1
            self.conversions[variant] += 1
        else:
            # if not converted, update the beta parameter
            self.failures[variant] += 1
        self.counts[variant] += 1

class EpsilonGreedy:
    # an algorithm for continuously balancing exploration vs. exploitation
    # epsilon: probability of exploration
    def __init__(self, n_variants, epsilon):
        self.n_variants = n_variants
        self.epsilon = epsilon
        self.counts = np.zeros(n_variants)
        self.conversions = np.zeros(n_variants)
        self.q_values = np.zeros(n_variants)
        self.alpha = 1  # for updating conversion rates

    def choose_variant(self):
        # Randomly choose between exploration and exploitation
        if np.random.random() < self.epsilon: # exploration
            return np.random.randint(self.n_variants)
        else: # greedy exploitation
            # Choose variant with highest conversion rate
            return np.argmax(self.conversions / self.counts)
        
    def update(self, variant, converted):
        self.counts[variant] += 1
        self.conversions[variant] += converted  

class TraditionalABTest:
    def __init__(self, n_variants):
        self.n_variants = n_variants
        self.counts = np.zeros(n_variants)
        self.conversions = np.zeros(n_variants)
        
    def choose_variant(self):
        # Even distribution for traditional A/B/C test
        return np.random.randint(self.n_variants)
    
    def update(self, variant, converted):
        self.counts[variant] += 1
        self.conversions[variant] += converted

# Experiment parameters
np.random.seed(42)
conversion_rates = [0.12, 0.14, 0.2]  # A, B, C variants
n_variants = len(conversion_rates)
visitors = 5000  # Total visitors to simulate
print_freq = 100  # Print results every n visitors
# Initialize components
abc_bandit = ABCTestBandit(conversion_rates)
thompson_strategy = ThompsonABCTest(n_variants)
epsilon_strategy = EpsilonGreedy(n_variants, 0.1)
traditional_strategy = TraditionalABTest(n_variants)

# Track allocations over time
thompson_allocations = np.zeros((visitors, n_variants))
epsilon_allocations = np.zeros((visitors, n_variants))
traditional_allocations = np.zeros((visitors, n_variants))

# Run experiment
for visitor in range(visitors):
    # Thompson Sampling approach
    variant = thompson_strategy.choose_variant()
    converted = abc_bandit.display_variant(variant)
    thompson_strategy.update(variant, converted)
    thompson_allocations[visitor, variant] = 1
    if visitor % print_freq == 0:
        print('thompson strategy:', 'visitor:', visitor, 'variant', variant, 'converted', converted)
    else:
        None

    # Epsilon Greedy approach
    variant = epsilon_strategy.choose_variant()
    converted = abc_bandit.display_variant(variant)
    epsilon_strategy.update(variant, converted)
    epsilon_allocations[visitor, variant] = 1

    if visitor % print_freq == 0:
        print('epsilon strategy:', 'visitor:', visitor, 'variant', variant, 'converted', converted)
    else:
        None
    
    # Traditional A/B/C test
    variant = traditional_strategy.choose_variant()
    converted = abc_bandit.display_variant(variant)
    traditional_strategy.update(variant, converted)
    traditional_allocations[visitor, variant] = 1
    if visitor % print_freq == 0:
        print('traditional strategy:', 'visitor:', visitor, 'variant', variant, 'converted', converted)
    else:
        None


# Sum up cumulatively 
cumulative_thompson = thompson_allocations.cumsum(axis=0)
cumulative_epsilon = epsilon_allocations.cumsum(axis=0)
cumulative_traditional = traditional_allocations.cumsum(axis=0)

# Modified print_results function to handle both strategies
def print_results(strategy, name):
    print(f"\n{name} Results:")
    for i in range(strategy.n_variants):
        cr = strategy.conversions[i] / strategy.counts[i]
        print(f"Variant {chr(65+i)}: Conversion Rate: {cr:.3f} ({strategy.conversions[i]}/{strategy.counts[i]})")

print_results(thompson_strategy, "Thompson Sampling")
print_results(epsilon_strategy, "Epsilon Greedy")
print_results(traditional_strategy, "Traditional A/B/C")

# Plot allocation percentages
plt.figure(figsize=(12, 6))
for i in range(n_variants):
    plt.plot(cumulative_thompson[:,i]/cumulative_thompson.sum(axis=1), 
             label=f'Variant {chr(65+i)} (Thompson)')
plt.title('Thompson Sampling Traffic Allocation')
plt.xlabel('Visitors')
plt.ylabel('Percentage Allocation')
plt.legend()
plt.grid(True)

plt.savefig('figures/thompson_ab_test_bandit.png')

plt.figure(figsize=(12, 6))
for i in range(n_variants):
    plt.plot(cumulative_epsilon[:,i]/cumulative_epsilon.sum(axis=1), 
             label=f'Variant {chr(65+i)} (Epsilon Greedy)')
plt.title('Epsilon Greedy Traffic Allocation')
plt.xlabel('Visitors')
plt.ylabel('Percentage Allocation')
plt.legend()
plt.grid(True)

plt.savefig('figures/epsilon_greedy_ab_test_bandit.png')

plt.figure(figsize=(12, 6))
for i in range(n_variants):
    plt.plot(cumulative_traditional[:,i]/cumulative_traditional.sum(axis=1), 
             label=f'Variant {chr(65+i)} (Traditional)')
plt.title('Traditional Test Traffic Allocation')
plt.xlabel('Visitors')
plt.ylabel('Percentage Allocation')
plt.legend()
plt.grid(True)

plt.savefig('figures/traditional_ab_test_bandit.png')