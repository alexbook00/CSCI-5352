import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

def ccdf_part_a(n, c, a):
    edges = np.zeros((n*c-6, 2)).astype(int) # structure is [out node, in node]
    in_degrees = np.zeros(n).astype(int)
    num_edges = 0

    for i in range(n):
        print(i)
        new_edges = []
        while len(new_edges) < min(c, i):
            r = np.random.uniform()
            if r < c / (c + a):
                target_i = np.random.randint(num_edges+1)
                target = edges[target_i][1]
            else:
                target = np.random.randint(i)
            if target not in new_edges:
                new_edges.append(target)
                edges[num_edges] = [i, target]
                in_degrees[target] += 1
                num_edges += 1

    min_degree = np.amin(in_degrees)
    max_degree = np.amax(in_degrees)
    bins = np.arange(0, max_degree+2, 1)
    hist, _ = np.histogram(in_degrees, bins)
    return bins[:-1], 1 - np.cumsum(hist)/np.sum(hist)

def ccdf_part_b(n, c, a):
    edges = np.zeros((n*c-6, 2)).astype(int) # structure is [out node, in node]
    in_degrees = np.zeros(n).astype(int)
    num_edges = 0

    for i in range(n):
        print(i)
        new_edges = []
        while len(new_edges) < min(c, i):
            r = np.random.uniform()
            if r < c / (c + a):
                target_i = np.random.randint(num_edges+1)
                target = edges[target_i][1]
            else:
                target = np.random.randint(i)
            if target not in new_edges:
                new_edges.append(target)
                edges[num_edges] = [i, target]
                in_degrees[target] += 1
                num_edges += 1
    return in_degrees

def ccdf_part_e(n, c):
    edges = np.zeros((n*c-6, 2)).astype(int) # structure is [out node, in node]
    in_degrees = np.zeros(n).astype(int)
    num_edges = 0

    for i in range(n):
        print(i)
        new_edges = []
        while len(new_edges) < min(c, i):
            target = np.random.randint(i)
            if (target not in new_edges) and (target != i):
                new_edges.append(target)
                edges[num_edges] = [i, target]
                in_degrees[target] += 1
                num_edges += 1

    min_degree = np.amin(in_degrees)
    max_degree = np.amax(in_degrees)
    bins = np.arange(0, max_degree+2, 1)
    hist, _ = np.histogram(in_degrees, bins)
    return bins[:-1], 1 - np.cumsum(hist)/np.sum(hist)

if __name__ == '__main__':
    ############################
    # PART A
    x1, y1 = ccdf_part_a(10**6, 3, 1)
    np.savetxt('data1x', x1, newline=" ")
    np.savetxt('data1y', y1, newline=" ")

    x2, y2 = ccdf_part_a(10**6, 3, 2)
    np.savetxt('data2x', x2, newline=" ")
    np.savetxt('data2y', y2, newline=" ")

    x3, y3 = ccdf_part_a(10**6, 3, 3)
    np.savetxt('data3x', x3, newline=" ")
    np.savetxt('data3y', y3, newline=" ")

    x4, y4 = ccdf_part_a(10**6, 3, 4)
    np.savetxt('data4x', x4, newline=" ")
    np.savetxt('data4y', y4, newline=" ")

    x1 = np.loadtxt('data1x')
    y1 = np.loadtxt('data1y')

    x2 = np.loadtxt('data2x')
    y2 = np.loadtxt('data2y')

    x3 = np.loadtxt('data3x')
    y3 = np.loadtxt('data3y')

    x4 = np.loadtxt('data4x')
    y4 = np.loadtxt('data4y')

    fig, ax = plt.subplots(1, 1)
    ax.plot(x1, y1)
    ax.plot(x2, y2)
    ax.plot(x3, y3)
    ax.plot(x4, y4)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('In-degree q')
    ax.set_ylabel('Fraction of nodes with in-degree of at least q')
    ax.set_xlim(10**0)
    fig.legend(['r = 1', 'r = 2', 'r = 3', 'r = 4'])
    plt.show()

    ############################
    # PART B
    first10percent = []
    last10percent = []
    for i in range(100):
        print(i)
        in_degrees_b = ccdf_part_b(10**6, 12, 5)
        avg_first_10percent = np.mean(in_degrees_b[:10**5])
        avg_last_10percent = np.mean(in_degrees_b[(10**6) - (10**5):])
        first10percent.append(avg_first_10percent)
        last10percent.append(avg_last_10percent)

    avg_first_10percent = np.mean(first10percent)
    avg_last_10percent = np.mean(last10percent)
    print(avg_first_10percent, avg_last_10percent)

    ############################
    # PART E
    x0_e, y0_e = ccdf_part_e(10**6, 3)
    np.savetxt('data0x_e', x0_e, newline=" ")
    np.savetxt('data0y_e', y0_e, newline=" ")

    x1_e, y1_e = ccdf_part_a(10**6, 3, 1)
    np.savetxt('data1x_e', x1_e, newline=" ")
    np.savetxt('data1y_e', y1_e, newline=" ")

    x4_e, y4_e = ccdf_part_a(10**6, 3, 4)
    np.savetxt('data4x_e', x4_e, newline=" ")
    np.savetxt('data4y_e', y4_e, newline=" ")

    x1_e = np.loadtxt('data1x_e')
    y1_e = np.loadtxt('data1y_e')

    x4_e = np.loadtxt('data4x_e')
    y4_e = np.loadtxt('data4y_e')

    x0_e = np.loadtxt('data0x_e')
    y0_e = np.loadtxt('data0y_e')

    fig, ax = plt.subplots(1, 1)
    ax.plot(x1_e, y1_e)
    ax.plot(x4_e, y4_e)
    ax.plot(x0_e, y0_e)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('In-degree q')
    ax.set_ylabel('Fraction of nodes with in-degree of at least q')
    fig.legend(['r = 1', 'r = 4', 'Uniform'])
    plt.show()
