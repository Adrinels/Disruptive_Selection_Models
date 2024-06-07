import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
population_size = 100
generations = 100
mutation_rate = 0.01
trait_mean_male = 10
trait_mean_female = 20
trait_std = 3
trait_bins = 30  # Number of bins for trait values
trait_range = (0, 30)  # Range of trait values

# Function to calculate fitness based on trait value
def calculate_fitness(trait, sex):
    if sex == "male":
        return np.exp(-(trait - trait_mean_male)**2 / (2 * trait_std**2))
    elif sex == "female":
        return np.exp(-(trait - trait_mean_female)**2 / (2 * trait_std**2))

# Function to perform selection and reproduction
def selection(population):
    fitness = np.array([calculate_fitness(individual['trait'], individual['sex']) for individual in population])
    selected_indices = np.random.choice(np.arange(len(population)), size=population_size, replace=True, p=fitness/fitness.sum())
    selected_individuals = [population[i] for i in selected_indices]
    return selected_individuals

def reproduction(selected_individuals):


    next_generation = []
    for individual in selected_individuals:
        trait = individual['trait'] + np.random.normal(loc=0, scale=mutation_rate)
        if trait < trait_range[0]:
            trait = trait_range[0]
        elif trait > trait_range[1]:
            trait = trait_range[1]
        next_generation.append({'trait': trait, 'sex': individual['sex']})
    return next_generation

def next_generation(population):
    selected_individuals = selection(population)
    return reproduction(selected_individuals)

# Initialize population
population = [{'trait': np.random.uniform(*trait_range), 'sex': np.random.choice(["male", "female"])} for _ in range(population_size)]

# Function to update plot for each generation
def update(frame):
    global population
    population = next_generation(population)
    traits_male = [individual['trait'] for individual in population if individual['sex'] == "male"]
    traits_female = [individual['trait'] for individual in population if individual['sex'] == "female"]
    ax.clear()
    ax.hist(traits_male, bins=np.linspace(*trait_range, num=trait_bins+1), alpha=0.5, color='blue', label='Male')
    ax.hist(traits_female, bins=np.linspace(*trait_range, num=trait_bins+1), alpha=0.5, color='red', label='Female')
    ax.set_xlabel('Trait Value')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Trait Distribution in Males and Females (Generation: {frame+1})')
    ax.legend()
    ax.set_xlim(*trait_range)  # Set fixed x-axis limits
    ax.set_ylim(0, population_size) 

    if frame == generations - 1:  # Terminate animation after reaching the specified number of generations
        ani.event_source.stop()

# Create figure and axis
fig, ax = plt.subplots()

# Animate
ani = FuncAnimation(fig, update, frames=generations, interval=100)
plt.show()
