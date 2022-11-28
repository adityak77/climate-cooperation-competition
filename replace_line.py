import os
import fileinput
import numpy as np

# only need to change below
dir_path = 'region_yamls'
prefix = 'xa_2'

def random_generator_a1():
    mu = 0.002
    sigma = 0.0007
    val = max(0, np.random.normal(mu, sigma))

    return val

def random_generator_a1_danger():
    mu = 0.5 # 0.002
    sigma = 0.2 # 0.0007
    val = max(0, np.random.normal(mu, sigma))

    return val

def random_generator_a2():
    mu = 0.002
    sigma = 0.0007
    val = max(0, np.random.normal(mu, sigma))

    return val

def random_generator_a2_danger():
    mu = 0.5 # 0.002
    sigma = 0.2 # 0.0007
    val = max(0, np.random.normal(mu, sigma))

    return val


generator_dicts = {
                    'xa_1' : random_generator_a1,
                    'xa_2' : random_generator_a2
                  }



random_generator = generator_dicts[prefix]

for i, fname in enumerate(os.listdir(dir_path)):
    if fname == 'default.yml':
        continue

    full_path = os.path.join(dir_path, fname)
    for line in fileinput.input(full_path, inplace=True):
        if line.strip().startswith(prefix):
            val = random_generator()
            # if i % 2:
            #     val = random_generator()
            # else:
            #     val = random_generator_a2_danger()
            print(f'  {prefix}: {val}')
        else:
            print(line.rstrip())

