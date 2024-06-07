import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


#Model based on slatkin model

# Parameters
initial_population_size = 200
generations = 200
mutation_rate = 0.001
trait_mean_male = -0.5
trait_mean_female = 0.5
trait_std = 0.1
maximum_crarring_capacity = 5
number_of_offsprng = 1
trait_bins = 40  # Number of bins for trait values
trait_range = (-1, 1)  # Range of trait values

# Function to calculate fitness based on trait value
def calculate_fitness(trait, sex):
    if sex == "male":
        return np.exp(-(trait - trait_mean_male)**2 / (2 * trait_std**2))
    elif sex == "female":
        return np.exp(-(trait - trait_mean_female)**2 / (2 * trait_std**2))

def carrying_capacity(z):
    return maximum_crarring_capacity * np.exp(- ((z-(trait_mean_female+trait_mean_male)/2)**2 / (2*trait_std**2) ))

def population_density(population):
    mean_trait = sum(individual['trait'] for individual in population) / len(population)
    K = carrying_capacity(mean_trait)

    return int ((number_of_offsprng*len(population)) / ( 1+ (number_of_offsprng-1)/K*len(population)))


# Function to perform selection and reproduction
def selection(population):
    fitness = np.array([calculate_fitness(individual['trait'], individual['sex']) for individual in population])
    density=population_density(population)
    selected_indices = np.random.choice(np.arange(len(population)), size=density, replace=True, p=fitness/fitness.sum())
    selected_individuals = [population[i] for i in selected_indices]
    return selected_individuals

def reproduction(selected_individuals):
    male_individuals = [individual for individual in selected_individuals if individual['sex'] == 'male']
    female_individuals = [individual for individual in selected_individuals if individual['sex'] == 'female']

    next_generation = []
    for _ in selected_individuals:
        #1/2 chance to have male or female
        chosen_list = random.choice([male_individuals, female_individuals])
        individual = np.random.choice(chosen_list)
        
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
population = [{'trait': 0, 'sex': np.random.choice(["male", "female"])} for _ in range(initial_population_size)]
global_data = {}

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
    ax.set_ylim(0, 100) 

    global global_data
    global_data[frame] = population.copy()

    if frame == generations - 1:  # Terminate animation after reaching the specified number of generations
        ani.event_source.stop()

# Create figure and axis
fig, ax = plt.subplots()

# Animate
ani = FuncAnimation(fig, update, frames=generations, interval=100)
plt.show()

plt.close()


#Create final graph
male_steps = []
female_steps= []
male_population = []
female_population= []

for k in global_data.keys():
    for i in global_data[k]:
        if i['sex'] == "male":
            male_steps.append(k)
            male_population.append(i['trait'])
        else:
            female_steps.append(k)
            female_population.append(i['trait'])

fig, ax = plt.subplots()

ax.hist2d(male_steps, male_population, bins = (np.arange(0, generations-1, 5), np.arange(-0.5, 0.5, 0.05)))

fig2, ax2 = plt.subplots()
ax2.hist2d(female_steps, female_population, bins = (np.arange(0, generations-1, 5), np.arange(-0.5, 0.5, 0.05)))


plt.show()
