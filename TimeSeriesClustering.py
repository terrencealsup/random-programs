"""
This algorithm takes in a data set of time-stamped data and an integer K.
It then clusters the data into K different time series according to some class of SDEs.

This program does the algorithm only for the simple class dX_t = dW_t i.e. Brownian motions.

By: Terrence Alsup
"""
import numpy as np
from sklearn.cluster import SpectralClustering
from matplotlib import pyplot as plt

"""
Transition probability for a Brownian motion, which follows a normal distribution.
@:param (x1,t1) is the initial point data and time
@:param (x2,t2) is the transition point data and time
@:param scale is a scale parameter that reduces the variance as 1/sqrt(scale)
@:return the transition probability
"""
def transition_probability(x1,t1, x2,t2, scale=1.0):
    variance = t2 - t1
    if variance == 0:
        if x1 == x2:
            return 1
        else:
            return 0
    return 1.0/np.sqrt(2.0*np.pi*variance*np.sqrt(scale)) * np.exp(-1.0*((x2-x1)**2)/(2.0*variance*np.sqrt(scale)))


# def transition_probability_OU((x1,t1),(x2,t2)):
#     d1 =



"""
Builds a transition probability graph for the simple model where
the transition probabilities are normal distributions as
governed by a Brownian motion.
@:param data = a list of the data points, assumes that the index corresponds to the time-stamp
@:param scale = a scaling paramter to reduce the variance as 1/sqrt(scale)
@:return a graph of all of the data points and the transition probabilities from all previous data points
"""
def build_transition_graph(data, scale):
    N = len(data)

    # The nodes are enumerated by their time-stamps.
    # Currently this uses the assumption that we do not have two data points with the same
    # time-stamp.  This will need to be fixed for any future practical use.
    nodes = range(N)

    # The edge list is a dictionary of edges to weights.
    edge_list = {}
    for i in range(2,N):
        for j in range(i-1):
            edge_list[(i,j)] = transition_probability(data[j],j,data[i],i, scale=scale)

    return (nodes, edge_list)


"""
G = (V, E) is a graph which will be the transition probability graph.
K is the number of clusters.
"""
def spectral_clustering(G, K):
    V,E = G
    N = len(V)
    # First construct the weight matrix.
    W = np.zeros((N,N))
    for edge in E.keys():
        i,j = edge
        W[i][j] = E[edge]
        W[j][i] = E[edge]

    # Construct the diagonal matrix.
    #D = np.diag(np.sum(W,1))

    # Compute the Laplacian matrix.
    #L = D - W

    # Compute the eigenvectors/eigenvalues for the Laplacian.
    #e_vals, e_vecs = np.linalg.eig(L)

    spec = SpectralClustering(n_clusters=K, affinity='precomputed')
    return spec.fit(W).labels_


"""
@:param data = a list of the data sorted by time
@:param labels = a list of cluster labels for each element in the data set
"""
def adjust_clusters(data, labels, K):
    N = len(data)
    # The last index where each data point occured.
    last_indices = [0]*K
    for i in range(1,N):
        likelihoods = np.zeros(K)
        for k in range(K):
            likelihoods[k] = transition_probability(data[last_indices[k]], last_indices[k], data[i],i)
        j = np.argmax(likelihoods)
        labels[i] = j
        last_indices[j] = i

    return labels





"""
For this test data we assume a simple model of K clusters.
size determines the total number of data points to generate.

We will randomly space the centers and include small perturbations to
the sample points based on a Brownian motion.

scale will scale down the variance.
"""
def generate_test_data(size, K, scale=1.0):
    data = np.zeros(size)
    labels = np.zeros(size)

    # A list of the last index where each data point in a time series occured.
    last_indices = [0]*K

    for i in range(size):
        # Randomly select a cluster.
        k = np.random.randint(K)
        # The centers are spaced as 0,10,20,...,10(K-1)
        if last_indices[k] == 0:
            center = 100*k
        else:
            center = data[last_indices[k]]
        data[i] = np.random.normal(center, i-last_indices[k]+1)/scale
        labels[i] = k
        last_indices[k] = i
    return data,labels

"""
@:param K = the number of distinct labels 0,...,K-1
@:param labels = the label of each data point
@:param data = the data provided
"""
def plot_data(data, labels, K, title='Data'):
    data = np.asarray(data)
    labels = np.asarray(labels)
    cstream = 'bgrcmykw'
    plt.figure()
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Data')
    plt.ylim(-100,400)
    for k in range(K):
        # Get all the data points in the k-th cluster.
        indices = np.where(labels==k)[0]
        plt.plot(indices, data[indices], c=cstream[k], label='Cluster %d'%k)
    plt.legend()



s = 2
data = generate_test_data(100, 5, scale=s)
G = build_transition_graph(data[0], scale=s)
l = spectral_clustering(G, 3)




plot_data(data[0],data[1],5,title='Test Data')
plot_data(data[0],l, 3, title='Algorithm Result')

l = adjust_clusters(data[0], l, 3)
plot_data(data[0],l,3,'Algorithm Result (Adjusted Clusters)')

plt.show()


"""
TO FIX/IMPLEMENT:

-does not work well with high variance

-should generate the test points using conditional distribution from previous point, not the starting point, will give
better approximations

-implement transition probabilities for OU process

-see if there is a numerical approach for finding transition probs for geometric brownian motion and any
general diffusion/Markov process

-try a greedy algorithm instead for assigning points to the time series where they have the highest likelihood,
if the model is Markov then this is computationally efficient since we only need to check the points immediately
preceeding each data point in time

-test the program for when the means are closer together and when the model is dynamic such as OU or GBM

"""
