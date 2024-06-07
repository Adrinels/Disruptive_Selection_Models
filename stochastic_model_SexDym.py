import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
from enum import Enum

#Model based on Deckmann and Doebeli 
#this model simulate more precisely the mating process
# Each Individuals now have locus simulating DNA
#When mating the child locus will be define accordingly to its parent's locus according to Mendelian Segregation without dominant or recesive allele

# Parameters
initial_population_size = 100
generations = 200
mutation_rate = 0.001
trait_mean_male = [0.2, -0.2]
trait_mean_female = [-0.2, 0.2]
trait_std = 0.25
maximum_crarring_capacity = 5
number_of_offsprng = 1
trait_bins = 20  # Number of bins for trait values
trait_range = (-1, 1)  # Range of trait values
shared_Loci = 22
total_Loci = 23


class Sex(Enum):
    MALE = "male"
    FEMALE = "female"

# Define the Individual dataclass
class Individual:
    sex: Sex
    locus: np.ndarray

    def __init__(self,sex,locus):
        self.sex=sex
        self.locus=locus
        self.genotype = np.sum(locus) / (2*total_Loci)


# Function to calculate fitness based on trait value
def calculate_fitness(trait, sex):
    if sex is Sex.MALE:
        return ( max([np.exp(-(trait - mean)**2 / (2 * trait_std**2)) for mean in trait_mean_male]))
    elif sex is Sex.FEMALE:
        return ( max([np.exp(-(trait - mean)**2 / (2 * trait_std**2)) for mean in trait_mean_female]))

def carrying_capacity(z):
    return maximum_crarring_capacity * np.exp(- ((z-(trait_mean_female[0]+trait_mean_male[0])/2)**2 / (2*trait_std**2) ))

def population_density(population):
    mean_trait = sum(individual.genotype for individual in population) / len(population)
    K = carrying_capacity(mean_trait)

    return int ((number_of_offsprng*len(population)) / ( 1+ (number_of_offsprng-1)/K*len(population)))


# Function to perform selection and reproduction
def selection(population):
    fitness = np.array([calculate_fitness(individual.genotype, individual.sex) for individual in population])
    density=population_density(population)
    selected_indices = np.random.choice(np.arange(len(population)), size=density, replace=True, p=fitness/fitness.sum())
    selected_individuals = [population[i] for i in selected_indices]
    return selected_individuals

def create_offspring(male, female):
    offspring_sex = np.random.choice([Sex.MALE, Sex.FEMALE])

    # Create the new locus array for the offspring
    offspring_locus = []
    for i, (locus_male, locus_female) in enumerate(zip(male.locus, female.locus)):
        #heritate entirely from same sex parent if not shaed
        if i >= shared_Loci:
            if offspring_sex is Sex.MALE:
                l1,l2 = locus_male
            else:
                l1,l2 = locus_female

            offspring_locus.append((mutate(l1),mutate(l2)))
            continue
    
        # Randomly choose one of the two alleles from each parent's locus
        allele_left = random.choice([locus_male, locus_female])
        allele_right = locus_male
        if allele_left is locus_male:
            allele_right = locus_female

        offspring_locus.append((mutate(allele_left[0]),mutate(allele_right[1])))

    offspring_locus = np.array(offspring_locus)

    # Create the new individual
    offspring = Individual(sex=offspring_sex, locus=offspring_locus)

    return offspring

def mutate(allele):
    if random.random() < mutation_rate:
        return -allele
    return allele


def reproduction(selected_individuals):
    male_individuals = [individual for individual in selected_individuals if individual.sex is Sex.MALE]
    female_individuals = [individual for individual in selected_individuals if individual.sex is Sex.FEMALE]
    next_generation = []
    for _ in selected_individuals:
        chosen_male = np.random.choice(male_individuals)
        chosen_female = np.random.choice(female_individuals)
        
        next_generation.append(create_offspring(chosen_male,chosen_female))
    return next_generation

def next_generation(population):
    selected_individuals = selection(population)
    return reproduction(selected_individuals)

# Initialize population
population = [Individual(sex=np.random.choice([Sex.MALE, Sex.FEMALE]), locus = np.full((total_Loci, 2), (-1, 1), dtype=np.float64)
 ) for _ in range(initial_population_size)]
global_data = {}

#mayor take driver
#distortion  amyotique

# Function to update plot for each generation
def update(frame):
    global population
    population = next_generation(population)
    traits_male = [individual.genotype for individual in population if individual.sex is Sex.MALE]
    traits_female = [individual.genotype for individual in population if individual.sex is Sex.FEMALE]
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
print("a")
plt.close()


#Create final graph
male_steps = []
female_steps= []
male_population = []
female_population= []

for k in global_data.keys():
    for i in global_data[k]:
        if i.sex is Sex.MALE:
            male_steps.append(k)
            male_population.append(i.genotype)
        else:
            female_steps.append(k)
            female_population.append(i.genotype)

fig, ax = plt.subplots()

ax.hist2d(male_steps, male_population, bins = (np.arange(0, generations-1, 5), np.arange(-0.5, 0.5, 0.05)))
ax.set_xlabel('Generation')
ax.set_ylabel('Trait')
ax.set_title(f'Trait progression for male)')
    

fig2, ax2 = plt.subplots()
ax2.hist2d(female_steps, female_population, bins = (np.arange(0, generations-1, 5), np.arange(-0.5, 0.5, 0.05)))
ax2.set_xlabel('Generation')
ax2.set_ylabel('Trait')
ax2.set_title(f'Trait progression for female)')

plt.show()
