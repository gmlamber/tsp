import numpy as np
import math
import itertools
import matplotlib.pyplot as plt
import matplotlib
import time as t
import sys
import csv

matplotlib.use('TkAgg')
sd = int(t.time())
np.random.seed(sd)

# Helper functions
def gen_locations3D(n):
    x = np.random.uniform(size=n)
    y = np.random.uniform(size=n)
    z = np.random.uniform(size=n)
    cities = np.column_stack((x, y ,z)) * 10.0
    return cities
    
def separation(p1, p2):
    return np.linalg.norm(p1 - p2)

def path_length(path, points):
    total = 0
    for i in range(len(path) - 1):
        total += separation(cities[path[i]], cities[path[i + 1]])
    return total

# TSP functions
def bf_otsp(cities):
    n = len(cities)
    min_length = float('inf')
    min_path = None
    
    perms = itertools.permutations(range(n))
    
    for perm in perms:
        length = path_length(perm, cities)
        if length < min_length:
            min_length = path_length(perm, cities)
            min_path = perm
            
    return min_path, min_length

def mc_otsp_swap(cities, steps=1000000, debug=False):
    n = len(cities)
    min_path = np.random.permutation(n)
    min_length = path_length(min_path, cities)
    
    if debug == True:
        print(f"starting path: {min_path}\nstarting length: {min_length}")
    
    for i in range(steps):
        new_path = min_path.copy()
        id1 = np.random.randint(0, n)
        id2 = np.random.randint(0, n - 1)
        id2 = (id2 + 1) % n if id2 >= id1 else id2
        
        new_path[id1], new_path[id2] = new_path[id2], new_path[id1]
        new_length = path_length(new_path, cities)
        
        if new_length < min_length:
            min_path = new_path
            min_length = new_length
            if debug == True:
                print(f"swap @ MC step {i}:\n\tpath: {min_path}\n\tlength: {min_length}")
            
    return min_path, min_length

if len(sys.argv) < 2 or len(sys.argv) > 3 :
    print("ERROR - USAGE: tsp.py <num_cities> [rand_seed]")
    sys.exit(0)
    
n = int(sys.argv[1])
if len(sys.argv) == 3:
    sd = int(sys.argv[2])
    np.random.seed(sd)
    
print(f"random seed: {sd}")
cities = gen_locations3D(n)

start = t.time()
bf_path, bf_length = bf_otsp(cities)
end = t.time()
bf_data = [bf_path, bf_length, end-start]
print(f"BF solution:\n\tmin path: {bf_path}\n\tmin length: {bf_length}\n\truntime: {end-start}s")
#swap_path, swap_length = mc_otsp_swap(cities)
#print(f"Swap solution:\n\tmin path: {swap_path}\n\tmin length: {swap_length}")
start = t.time()
swap_path, swap_length = mc_otsp_swap(cities,steps=math.factorial(n)+1)
end = t.time()
swap_data = [swap_path, swap_length, end-start]
print(f"MC solution:\n\tmin path: {swap_path}\n\tmin length: {swap_length}\n\truntime: {end-start}s")

fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(cities[:,0], cities[:,1], cities[:,2])

bfpath_x = [cities[:,0][i] for i in bf_path]
bfpath_y = [cities[:,1][i] for i in bf_path]
bfpath_z = [cities[:,2][i] for i in bf_path]
ax1.plot(bfpath_x, bfpath_y, bfpath_z, color='red')
ax1.set_title(f"Brute-Force Path; Length={round(bf_length, 3)}")

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(cities[:,0], cities[:,1], cities[:,2])

mcpath_x = [cities[:,0][i] for i in swap_path]
mcpath_y = [cities[:,1][i] for i in swap_path]
mcpath_z = [cities[:,2][i] for i in swap_path]
ax2.plot(mcpath_x, mcpath_y, mcpath_z, color='green')
ax2.set_title(f"Monte Carlo Path; Length={round(swap_length, 3)}")

manager = plt.get_current_fig_manager()
manager.window.state('zoomed')

plt.savefig(f"{n}tsp_{sd}.png", format='png')

with open('data.csv', 'a') as f:
    writer = csv.writer(f)
    row = [sd]
    row.extend(bf_data)
    row.extend(swap_data)
    
    writer.writerow(row)
    
