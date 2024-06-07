import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
population_size = 100
generations = 100
mutation_rate = 0.2
optimums = [10,20]
trait_std = 3

#Graph option
HIST_BINS = np.linspace(-4, 4, 100)
trait_bins = 30
trait_range = (0, 30)

#Initial Population
# Initialize population
uniform_population = [{'trait': 15} for _ in range(population_size)]
random_population = [{'trait': np.random.uniform((8,12))} for _ in range(population_size)]

population = uniform_population



# Function to calculate fitness based on trait value
def calculate_fitness(trait):
    fitness = 0
    for o in optimums :
        fitness += np.exp(-(trait - o)**2 / (2 * trait_std**2))  
    return fitness

# Function to perform selection and reproduction
def selection(population):
    #selection
    fitness = np.array([calculate_fitness(individual['trait']) for individual in population])
    selected_indices = np.random.choice(np.arange(len(population)), size=population_size, replace=True, p=fitness/fitness.sum())
    selected_individuals = [population[i] for i in selected_indices]
    
    return selected_individuals
#reproduction
def reproduction(selcted_individuals):
    next_generation = []
    for individual in selcted_individuals:
        trait = individual['trait'] + np.random.normal(loc=0, scale=mutation_rate)
        if trait < 0:
            trait = 0
        elif trait > 30:
            trait = 30
        next_generation.append({'trait': trait })
    return next_generation

def next_generation(population):
    selcted_individuals= selection(population)
    return reproduction(selcted_individuals)

# Function to update plot for each generation
def update(frame):
    global population
    population = next_generation(population)
    traits = [individual['trait'] for individual in population]
    ax.clear()
    ax.hist(traits, bins=np.linspace(*trait_range, num=trait_bins+1), alpha=0.5, color='blue', label='Individual')

    ax.set_xlabel('Trait Value')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Trait Distribution in Males and Females (Generation: {frame+1})')
    ax.set_xlim(0, 30)  # Set fixed x-axis limits
    ax.set_ylim(0, population_size)
    ax.legend()
    if frame == generations - 1:  
        ani.event_source.stop()

# Create figure and axis
fig, ax = plt.subplots()
plt.xlim(0, 30)


# Animate
ani = FuncAnimation(fig, update, frames=generations, interval=100)
plt.show()
