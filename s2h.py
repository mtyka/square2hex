import numpy as np
import math
import copy
import sys
import matplotlib.pyplot as plt
from munkres import Munkres

bestdev = 1.0
bestNs = None
maxN=20
limitTotal=int(sys.argv[1]) #530
for A in range(2,maxN):
  for C in range(2,maxN):
    for B in range(2,maxN):
      for D in range(2,maxN):
        NSx=A*C
        NSy=B*D
        NHx=A*B
        NHy=C*D
        #print A,B,C,NSx, NSy, NHx, NHy
        assert NSx*NSy == NHx*NHy
        devS = NSx*1.0/NSy
        devH = (NHx*1.0/NHy)/math.sqrt(3.0/4)
        dev = abs(math.log(devS/devH))
        if NSx*NSy > limitTotal: continue
        if dev < bestdev:
          bestdev = dev
          bestNs = (NSx, NSy, NHx, NHy)
          #print "Deviations ", devS, devH, dev 

NSx, NSy, NHx, NHy = bestNs
N=NSx*NSy
#print bestNs, bestdev

xaxis=np.linspace(0, 1.0-1.0/NSx, NSx)
yaxis=np.linspace(0, 1.0-1.0/NSy, NSy)
square = np.vstack(np.meshgrid(xaxis,yaxis)).reshape(2,-1).T

xaxis=np.linspace(0, 1.0-1.0/NHx, NHx)
yaxis=np.linspace(0, 1.0-1.0/NHy, NHy)
xv,yv = np.meshgrid(xaxis,yaxis)
for i in range(0,NHy,2):
  xv[i,:] += 0.5/NHx
hexa = np.vstack([xv,yv]).reshape(2,-1).T
#print square.shape, hexa.shape, N

#plt.scatter(hexa[:,0],hexa[:,1])
#plt.scatter(hexa[:,0],hexa[:,1]+1.0)
#plt.scatter(hexa[:,0]+1.0,hexa[:,1])
#plt.scatter(square[:,0]+3.0,square[:,1])
#plt.scatter(square[:,0]+3.0,square[:,1]+1.0)
#plt.scatter(square[:,0]+4.0,square[:,1])
#plt.show()

def make_pair_periodic(p1,p2):
  if p1[0] - p2[0] >  0.5: p2[0] += 1.0
  if p1[0] - p2[0] < -0.5: p2[0] -= 1.0
  if p1[1] - p2[1] >  0.5: p2[1] += 1.0
  if p1[1] - p2[1] < -0.5: p2[1] -= 1.0
  return p2

dists = np.zeros((N,N))
for i in range(N):
  p1 = hexa[i]
  for j in range(0, N):
    p2 = copy.copy(square[j])
    p2 = make_pair_periodic(p1,p2)
    sqrdist = np.sum((p1-p2)**2)
    dists[i,j] = sqrdist
matrix = []
for i in range(N):
  matrix.append(list(dists[i,:]))

m = Munkres()
indexes = m.compute(matrix)
total = 0.0
for row, column in indexes:
  hexa[row] = make_pair_periodic(square[column], hexa[row]) 
  print square[column][0], square[column][1], hexa[row][0], hexa[row][1]
  value = matrix[row][column]
  total += value
  #print '%s -> %s  %f' % (str(square[column]), str(hexa[row]), value)

#print "total cost: ", total
#plt.margins(0.05)
#for row, column in indexes:
#  print square[column][0], square[column][1], hexa[row][0], hexa[row][1]
#  plt.plot([square[column][0], hexa[row][0]], 
#           [square[column][1], hexa[row][1]], 'k-')
#plt.show()


