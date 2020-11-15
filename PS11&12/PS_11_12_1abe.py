import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

def ccdf(n, c, a):
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
    return in_degrees

if __name__ == '__main__':
    ############################
    # PART A
    in_degrees_1 = ccdf(10**6, 3, 1)
    np.savetxt('data1', in_degrees_1, newline=" ")
    in_degrees_2 = ccdf(10**6, 3, 2)
    np.savetxt('data2', in_degrees_2, newline=" ")
    in_degrees_3 = ccdf(10**6, 3, 3)
    np.savetxt('data3', in_degrees_3, newline=" ")
    in_degrees_4 = ccdf(10**6, 3, 4)
    np.savetxt('data4', in_degrees_4, newline=" ")
    in_degrees_1 = np.loadtxt('data1').astype(int)
    in_degrees_2 = np.loadtxt('data2').astype(int)
    in_degrees_3 = np.loadtxt('data3').astype(int)
    in_degrees_4 = np.loadtxt('data4').astype(int)

    fig, ax = plt.subplots(1, 1)
    sns.ecdfplot(in_degrees_1, complementary=True, log_scale=[True, True], legend=True)
    sns.ecdfplot(in_degrees_2, complementary=True, log_scale=[True, True], legend=True)
    sns.ecdfplot(in_degrees_3, complementary=True, log_scale=[True, True], legend=True)
    sns.ecdfplot(in_degrees_4, complementary=True, log_scale=[True, True], legend=True)
    ax.set_xlabel('In-degree q')
    ax.set_ylabel('Fraction of nodes with in-degree of at least q')
    fig.legend(['r = 1', 'r = 2', 'r = 3', 'r = 4'])
    plt.show()

    ############################
    # PART B
    in_degrees_b = ccdf(10**6, 12, 5)
    np.savetxt('data_b', in_degrees_b, newline=" ")
    in_degrees_b = np.loadtxt('data_b').astype(int)

    first10percent = []
    last10percent = []
    for i in range(100):
        print(i)
        in_degrees_b = ccdf(10**6, 12, 5)
        avg_first_10percent = np.mean(in_degrees_b[:10**5])
        avg_last_10percent = np.mean(in_degrees_b[(10**6) - (10**5):])
        first10percent.append(avg_first_10percent)
        last10percent.append(avg_last_10percent)

    avg_first_10percent = np.mean(first10percent)
    avg_last_10percent = np.mean(last10percent)
    print(avg_first_10percent, avg_last_10percent)

    ############################
    # PART E
    in_degrees_e = ccdf_part_e(10**6, 3)
    np.savetxt('data_e', in_degrees_e, newline=" ")
    in_degrees_1 = ccdf(10**6, 3, 1)
    np.savetxt('data1', in_degrees_1, newline=" ")
    in_degrees_4 = ccdf(10**6, 3, 4)
    np.savetxt('data4', in_degrees_4, newline=" ")
    in_degrees_e = np.loadtxt('data_e').astype(int)
    in_degrees_1 = np.loadtxt('data1').astype(int)
    in_degrees_4 = np.loadtxt('data4').astype(int)

    fig, ax = plt.subplots(1, 1)
    sns.ecdfplot(in_degrees_1, complementary=True, log_scale=[True, True], legend=True)
    sns.ecdfplot(in_degrees_4, complementary=True, log_scale=[True, True], legend=True)
    sns.ecdfplot(in_degrees_e, complementary=True, log_scale=[True, True], legend=True)
    ax.set_xlabel('In-degree q')
    ax.set_ylabel('Fraction of nodes with in-degree of at least q')
    fig.legend(['r = 1', 'r = 4', 'Uniform'])
    plt.show()
