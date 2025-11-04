from functools import partial
from random import choices, randint, random, randrange
import time
from typing import List, Callable, Tuple
from collections import namedtuple

Genome = List[int]
Population = List[Genome]
FitnessFunc = Callable[[Genome], int]
PopulateFunc = Callable[[], Population]  # takes nothing, gives new solutions
SelectionFunc = Callable[[Population, FitnessFunc], Tuple[Genome, Genome]]  # takes population, fitness to select 2 solutions to be the parents
CrossoverFunc = Callable[[Genome, Genome], Tuple[Genome, Genome]]  # takes 2 genomes, returns 2 new genomes
MutationFunc = Callable[[Genome], Genome] # takes 1 genome and sometimes returns a modified one
Thing = namedtuple('Thing', ['name', 'value', 'weight'])

# data
things = [
  Thing('Laptop', 500, 2200),
  Thing('Headphones', 150, 160),
  Thing('Coffee Mug', 60, 350),
  Thing('Notepad', 40, 333),
  Thing('Water Bottle', 30, 192),
]

more_things = [
  Thing('Mints', 5, 25),
  Thing('Socks', 10, 38),
  Thing('Tissues', 15, 80),
  Thing('Phone', 500, 200),
  Thing('Baseball Cap', 100, 70),
] + things

# genetic representation of the solution
def generate_genome(length: int) -> Genome:
  return choices([0, 1], k = length)

# generate new solutions
def generate_population(size: int, genome_length: int) -> Population:
  return [generate_genome(genome_length) for _ in range(size)]

# fitness function to evaluate solutions
def fitness(genome: Genome, things: [Thing], weight_limit: int) -> int:
  if(len(genome) != len(things)):
    raise ValueError("Genome and things must be of the same length")
  
  weight = 0
  value = 0

  for i, thing in enumerate(things):
    if genome[i] == 1:
      weight += thing.weight
      value += thing.value

      if weight > weight_limit:
        return 0
      
  return value

# selection - pair of solutions which will be the parents of two new solutions for the next generation
def selection_pair(population: Population, fitness_func: FitnessFunc) -> Population:
  return choices(
    population = population,
    weights = [fitness_func(genome) for genome in population],
    k = 2
  )

# crossover - randomly cut genomes in half and combine 
def single_point_crossover(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
  if len(a) != len(b):
    raise ValueError("Genomes must be of the same length")
  
  length = len(a)
  if length < 2:
    return a, b

  p = randint(1, length -1)
  return a[0:p] + b [p:], b[0:p] + a[p:]  # first part of a, second of b, and vice versa


# mutation
def mutation(genome: Genome, num: int = 1, probability: float = 0.5) -> Genome:
  for _ in range(num):
    index = randrange(len(genome))
    genome[index] = genome[index] if random() < probability else abs(genome[index] - 1)
  return genome

# evolution
def run_evolution(
    populate_func: PopulateFunc,
    fitness_func: FitnessFunc,
    fitness_limit: int,  # if the fitness of the best soltuion exceeds the limit, it's done
    selection_func: SelectionFunc = selection_pair,
    crossover_func: CrossoverFunc = single_point_crossover,
    mutation_func: MutationFunc = mutation,
    generation_limit: int = 100  # max number of generationg the evolution runs for
) -> Tuple[Population, int]:
  # generate the first generation
  population = populate_func()

  # loop for generation limit times
  for i in range(generation_limit):
    # sort population by fitness - top solutions populate first indexes of the genome
    population = sorted(
      population,
      key = lambda genome: fitness_func(genome),
      reverse = True
    )

    if fitness_func(population[0]) >= fitness_limit:
      break

    next_generation = population[0:2]

    # generate all new solutions for the next generation
    for j in range(int(len(population) / 2) - 1):
      parents = selection_func(population, fitness_func)
      offspring_a, offspring_b = crossover_func(parents[0], parents[1])
      offspring_a = mutation_func(offspring_a)
      offspring_b = mutation_func(offspring_b)
      next_generation += [offspring_a, offspring_b]

    population = next_generation

  population = sorted(
    population,
    key = lambda genome: fitness_func(genome),
    reverse = True
  )

  return population, i

start = time.perf_counter()
population, generations = run_evolution(
  populate_func = partial(
    generate_population, size = 10, genome_length = len(things)
  ),
  fitness_func = partial(
    fitness, things = things, weight_limit = 3000
  ),
  fitness_limit = 740,
  generation_limit = 100
)
end = time.perf_counter()

# print things based on genome
def genome_to_things(genome: Genome, things: [Thing]) -> Thing:
  result = []
  for i, thing in enumerate(things):
    if genome[i] == 1:
      result += [thing.name]

  return result


print(f"Number of generations: {generations}")
print(f"Time: {end - start}s")
print(f"Best solution: {genome_to_things(population[0], things)}")