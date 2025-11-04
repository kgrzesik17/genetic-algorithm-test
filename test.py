from collections import namedtuple

Thing = namedtuple('Thing', ['value', 'weight'])

knapsack = []  # item_count | max_weight
things = []  # value | weight

file = open("low-dimensional/f1_l-d_kp_10_269", "r")

for i, line in enumerate(file):
  line = line.strip()
  line = line.split()

  if i == 0:
    knapsack.append(Thing(int(line[0]), int(line[1])))
    continue

  things.append(Thing(int(line[0]), int(line[1])))


print(knapsack)
print(things)




things1 = [
  Thing(500, 2200),
  Thing(150, 160),
  Thing(60, 350),
  Thing(40, 333),
  Thing(30, 192),
]

# print(things1)