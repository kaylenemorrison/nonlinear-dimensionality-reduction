from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import connected_components
import numpy as np
import math
import random
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import linalg as LA
from mpl_toolkits import mplot3d
from sklearn.decomposition import PCA
import nltk
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
nltk.download('punkt')

def summation_component(yi,yj,pi):
  if (np.array_equal(yi,yj)):
    return np.zeros((yi.shape[0]))
  else:
    sub = np.subtract(yi,yj)### (yi - yj)
    norm = LA.norm(sub) ### || yi - yj ||
    div = np.divide(sub, norm)## (yi - yj) / || yi - yj ||
    pi_term = 2* (norm - pi) ## 2* (|| yi - yj || - pi)
    total = np.multiply(pi_term, div )
    return total

def objective_function_component(yi,yj,pi):
  sub = np.subtract(yi,yj)
  norm = LA.norm(sub)
  total = (norm - pi) **2 ### (||(yi - yj)|| - pi)^2
  return total

def minimize(pi_matrix, target_dimension, learning_rate, iterations):
  num_vectors = len(pi_matrix)
  y = np.random.rand(num_vectors, target_dimension) ## 4000 x 2 matrix
  for iteration in range(iterations):
    dif = 0
    for i in range(len(y)): ## for each vector
      yi = y[i]
      for j in range(len(y)):
        yj = y[j]
        grad_addition = summation_component(yi,yj, pi_matrix[i][j]) ## compute component of summation 
        dif += objective_function_component(yi,yj, pi_matrix[i][j])
        y[i] = np.subtract(yi, np.multiply(grad_addition, learning_rate))  ## yi = yi - learn_rate * grad_yi
    print("function_value: ", dif) 
  return y

def main():
  with open('../swiss_roll.txt') as f:
    swiss_roll = f.readlines()

  x= []
  y = []
  z = []

  for vector in swiss_roll:
    vector = nltk.word_tokenize(vector)
    for i in range(len(vector)):
      vector[i] = float(vector[i])
    x.append(vector[0]) 
    y.append(vector[1])
    z.append(vector[2])
 

  fig = plt.figure()
 
  # syntax for 3-D projection
  ax = plt.axes(projection ='3d')
  ax.scatter(x, y, z, marker = 'o')
  ax.set_title('swiss_roll')
  plt.show()

  SwissRoll = np.zeros((len(x),3))
  for i in range(len(x)):
    SwissRoll[i][0] = x[i]
    SwissRoll[i][1] = y[i]
    SwissRoll[i][2] = z[i]

  pca = PCA(n_components=2)
  trans_SwissRoll = pca.fit_transform(SwissRoll, y=None)
  x_2d = []
  y_2d = []
  for row in trans_SwissRoll:
    x_2d.append(row[0])
    y_2d.append(row[1])
  
  plt.scatter(x_2d,y_2d)
  plt.title("Swiss Roll PCA")
  plt.show()

  k = 10
  KNN_graph = kneighbors_graph(X = SwissRoll, n_neighbors = k, mode='distance', include_self=False)
  dist_matrix = dijkstra(csgraph = KNN_graph, directed = True, min_only = False)

  for i in range(len(dist_matrix)):
    for j in range(len(dist_matrix)):
      if (np.isnan(dist_matrix[i][j])):
        print("nan value")

  y = minimize(dist_matrix, 2, 0.2, 10)

  v1 = []
  v2 = []
  for row in y:
    print(row)
    v1.append(row[0])
    v2.append(row[1])

  plt.scatter(v1,v2)
  plt.title("Swiss Role Nonlinear Dimensionality Reduction")
  plt.show()

  with open('../swiss_roll_hole.txt') as f:
    swiss_roll_hole = f.readlines()

  x= []
  y = []
  z = []
  for vector in swiss_roll_hole:
    vector = nltk.word_tokenize(vector)
    for i in range(len(vector)):
      vector[i] = float(vector[i])
    x.append(vector[0]) 
    y.append(vector[1])
    z.append(vector[2])
  fig = plt.figure()
 
# syntax for 3-D projection
  ax = plt.axes(projection ='3d')
  ax.scatter(x, y, z, marker = 'o')
  ax.set_title('swiss_roll_holl')
  plt.show()

  SwissRollHole = np.zeros((len(x),3))
  for i in range(len(x)):
    SwissRollHole[i][0] = x[i]
    SwissRollHole[i][1] = y[i]
    SwissRollHole[i][2] = z[i]

  pca = PCA(n_components=2)
  trans_SwissRollHole = pca.fit_transform(SwissRollHole, y=None)
  x_2d = []
  y_2d = []
  for row in trans_SwissRollHole:
    x_2d.append(row[0])
    y_2d.append(row[1])
  
  plt.scatter(x_2d,y_2d)
  plt.title("Swiss Roll Hole PCA")
  plt.show()

  k = 10
  KNN_graph_2 = kneighbors_graph(X = SwissRollHole, n_neighbors = k, mode='distance', include_self=False)
  dist_matrix_2 = dijkstra(csgraph = KNN_graph_2, directed = True, min_only = False)

  y = minimize(dist_matrix_2, 2, 0.15, 10)

  v1 = []
  v2 = []
  for row in y:
    print(row)
    v1.append(row[0])
    v2.append(row[1])

  plt.scatter(v1,v2)
  plt.title("Swiss Roll Hole Nonlinear Dimensionality Reduction")
  plt.show()

if __name__ == "__main__":
    main()


