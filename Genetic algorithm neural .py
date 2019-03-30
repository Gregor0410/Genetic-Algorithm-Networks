import numpy as np
import random
def fitness(x,trainingData):
    fitnesses = []
    for i in x:
        n = Network(i)
        fitness = 0
        for example in trainingData:
            fitness += -(abs(n.feedforward(example[0])-example[1]))
        fitnesses.append(fitness[0,0])
    return fitnesses
def getParents(noOfParents,population,fitness):
    parents = []
    for i in range(noOfParents):
        idx = np.argmax(fitness)
        parents.append(population[idx])
        fitness[idx] = -9999999
    return parents
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))
def getNumWeightsAndBiases(sizes):
    numBiases = sum([i for i in sizes[1:]])
    numWeights =sum([x*y for x,y in zip(sizes[:-1],sizes[1:])])
    return numWeights,numBiases
def mutate(offspringCrossover,mutationSize,sizes):
    for i in range(len(offspringCrossover)):
        numWeights,numBiases = getNumWeightsAndBiases(sizes)
        weightMutationPoint = random.randint(0,numWeights-1)
        biasMutationPoint = random.randint(0,numBiases-1)
        j = 0
        for k in range(len(offspringCrossover[i][0])):
            for l in range(len(offspringCrossover[i][0][k])):
                if j == weightMutationPoint:
                        offspringCrossover[i][0][k][l] += random.gauss(0,mutationSize)
                j+=1
        j = 0
        for k in range(len(offspringCrossover[i][1])):
            for l in range(len(offspringCrossover[i][1][k])):
                if j == biasMutationPoint:
                        offspringCrossover[i][1][k][l] += random.gauss(0,mutationSize)
                j+=1
        return offspringCrossover
def crossover(parents,noOfOffspring,sizes):
    numWeights,numBiases = getNumWeightsAndBiases(sizes)
    offspring = emptyGenePool(noOfOffspring,sizes)
    for i in range(noOfOffspring):
        parent1 = parents[i%len(parents)]
        parent2 = parents[(i+1)%len(parents)]
        weightCrossoverPoint = random.randint(0,numWeights-1)
        biasCrossoverPoint = random.randint(0,numBiases-1)
        j = 0
        for k in range(len(offspring[i][0])):
            for l in range(len(offspring[i][0][k])):
                if j < weightCrossoverPoint:
                        offspring[i][0][k][l] = parent1[0][k][l]
                else:
                    offspring[i][0][k][l] = parent2[0][k][l]
                j+=1
        j = 0
        for k in range(len(offspring[i][1])):
            for l in range(len(offspring[i][1][k])):
                if j < biasCrossoverPoint:
                        offspring[i][1][k][l] = parent1[1][k][l]
                else:
                    offspring[i][1][k][l] = parent2[1][k][l]
                j+=1
    return offspring

class Network:
    def __init__(self,genes):
        self.genes = genes
        self.score = 0
    def feedforward(self,a):
        for b,w in zip(self.genes[1], self.genes[0]):
            a = sigmoid(np.dot(w,a)+b)
        return a
def emptyGenePool(size,sizes):
    genePool = []
    for i in range(size):
        genePool.append([[np.zeros((y,x)) for x,y in zip(sizes[:-1],sizes[1:])],[np.zeros((y,1)) for y in sizes[1:]]])
    return genePool
def randomGenePool(size,sizes):
    genePool = []
    for i in range(size):
        genePool.append([[np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])],[np.random.randn(y,1) for y in sizes[1:]]])
    return genePool

def newGeneration(population, sizes,fitnesses, parentsPerGen,populationSize,mutationSize):
    parents = getParents(parentsPerGen, population, fitnesses)
    offspringCrossover = crossover(parents, populationSize - len(parents),sizes)
    offspringMutate = mutate(offspringCrossover, mutationSize, sizes)
    population[:parentsPerGen] = parents
    population[parentsPerGen:] = offspringMutate
    return population
def genesFeedforward(genes,x):
    n =  Network(genes)
    return n.feedforward(x)
sizes = [1,1]
trainingData = [([[1]],[[0]]),([[0]],[[1]])]
population = randomGenePool(10,sizes)
for i in range(100):
    population = newGeneration(population, sizes, fitness(population,trainingData), 2, 10, 1)
print(genesFeedforward(population[0],[[1]]))
print(genesFeedforward(population[0],[[0]]))
