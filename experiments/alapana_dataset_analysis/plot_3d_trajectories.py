import matplotlib.pyplot as plt

import random

fig = plt.figure()
ax = plt.axes(projection ='3d')


feature='3dvelocityDTWHand'

i = 736#random.choice(all_groups['index'].values)

i_vec = index_features[i][feature]

# plotting
ax.plot3D(i_vec[:,0], i_vec[:,1], i_vec[:,2], c='blue', label=f'index={i}')

plt.legend()
#ax.set_ylim((-10,10))
#ax.set_xlim((-10,10))
#ax.set_zlim((-10,10))
ax.set_ylabel('y')
ax.set_xlabel('x')
ax.set_zlabel('z')
plt.savefig('3d_plot.png')
plt.close('all')