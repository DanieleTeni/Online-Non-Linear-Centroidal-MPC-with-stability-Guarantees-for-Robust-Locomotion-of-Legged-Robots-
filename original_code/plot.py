import numpy as np
import matplotlib.pyplot as plt
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
dati = np.loadtxt(os.path.join(current_dir,'lip.txt'))
dati = dati[:-40]
print(len(dati))
lipx = dati[:, 0]
lipy = dati[:, 1]
lipz = dati[:, 2]

t = np.arange(len(lipx)) 

data2 = np.loadtxt(os.path.join(current_dir,'cuhw.txt'))
data2 = data2[:-2]
hx = data2[:, 0]
hy = data2[:, 1]
hz = data2[:, 2]

data3 = np.loadtxt(os.path.join(current_dir,'hw_des.txt'))
data3 = data3[:-2]
print(len(data3))
hx1 = data3[:, 0]
hy1 = data3[:, 1]
hz1 = data3[:, 2]


show_hw1 = False  


plt.figure(figsize=(10, 8))

# hx
plt.subplot(3, 1, 1)
plt.plot(t, lipx, color='r', label='lip')
plt.plot(t, hx, color='b', label='centroidal')
if show_hw1:  
    plt.plot(t, hx1, color='g', label='output mpc')
plt.title('x component')
plt.ylabel('hx')
plt.grid(True)
plt.legend()

# hy
plt.subplot(3, 1, 2)
plt.plot(t, lipy, color='r', label='lip')
plt.plot(t, hy, color='b', label='centroidal')
if show_hw1:  
    plt.plot(t, hy1, color='g', label='output mpc')
plt.title('y component')
plt.ylabel('hy')
plt.grid(True)
plt.legend()

# hz
plt.subplot(3, 1, 3)
plt.plot(t, lipz, color='r', label='lip')
plt.plot(t, hz, color='b', label='centroidal')
if show_hw1:  
    plt.plot(t, hz1, color='g', label='output mpc')
plt.title('z component')
plt.xlabel('Time')
plt.ylabel('hz')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
