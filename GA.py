import numpy as np
import random
def fitness(x):
    return [sum(i) for i in x]
def getParents(noOfParents,population,fitness):
    parents = np.empty((noOfParents,population.shape[1]))
    for i in range(noOfParents):
        idx = np.argmax(fitness)
        parents[i,:] = population[idx,:]
        fitness[idx] = -9999999
    return parents
def crossover(parents,noOfOffspring):
    offspring = np.empty((noOfOffspring,parents.shape[1]))
    crossoverPoint = int(offspring.shape[1]/2)
    for i in range(noOfOffspring):
        parent1 = parents[i%parents.shape[0]]
        parent2 = parents[(i+1)%parents.shape[0]]
        offspring[i,:crossoverPoint] = parent1[:crossoverPoint]
        offspring[i,crossoverPoint:] = parent2[crossoverPoint:]
    return offspring
def mutation(offspringCrossover):
    for i in range(offspringCrossover.shape[0]):
        offspringCrossover[random.randint(0,offspringCrossover.shape[0]-1)] += random.gauss(0,1)
    return offspringCrossover
pop_shape = (10,5)
population = np.random.uniform(low=-1.0, high=1.0,size=pop_shape)
def newGeneration(population,parentsPerGen,fitnesses):
    fitnesses = fitness(population)
    parents = getParents(parentsPerGen,population,fitnesses)
    offspringCrossover = crossover(parents,pop_shape[0]-parents.shape[0])
    offspringMutation = mutation(offspringCrossover)
    population[:parents.shape[0],:] =parents
    population[parents.shape[0]:,:] = offspringMutation
    return population
