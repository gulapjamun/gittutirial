import numpy as np
import matplotlib.pyplot as plt
t_pi=2
H_ic = np.full((3, 3), t_pi, dtype=complex)
H_ic = H_ic - t_pi * np.identity(3)
print(H_ic)




def Hamil_k(t_pi, k_1, k_2, delta=1):
    pi = np.pi

    # H_incell
    H_ic = np.full((3, 3), t_pi, dtype=complex)
    H_ic = H_ic - t_pi * np.identity(3)

    # H_A
    H_z = np.zeros((3, 3), dtype=complex)
    H_1 = H_z.copy()
    H_1[0, 1] = t_pi
    H_2 = H_z.copy()
    H_2[0, 2] = t_pi
    H_A = np.exp(-2 * pi * (1.0j) * k_1) * H_1 + np.exp(-2 * pi * (1.0j) * (k_1 + k_2)) * H_2

    # H_B
    H_1 = H_z.copy()
    H_1[1, 0] = t_pi
    H_2 = H_z.copy()
    H_2[1, 2] = t_pi
    H_B = np.exp(2 * pi * (1.0j) * k_1) * H_1 + np.exp(-2 * pi * (1.0j) * (k_2)) * H_2

    # H_C
    H_1 = H_z.copy()
    H_1[2, 0] = t_pi
    H_2 = H_z.copy()
    H_2[2, 1] = t_pi
    H_C = np.exp(2 * pi * (1.0j) * (k_1 + k_2)) * H_1 + np.exp(2 * pi * (1.0j) * (k_2)) * H_2

    # H_total
    H = H_ic + H_A + H_B + H_C
    H[0, 0] += delta
    H[1, 1] -= delta
    return H

x_list = np.arange(0, 1.01, 0.002)
EE_total=[]

for x in x_list:
    H_x=Hamil_k(1.0,x,x,1.0)
    EE,EV=np.linalg.eigh(H_x)
    EE_total.append(EE)
EE_total=np.array(EE_total)



plt.plot(x_list, EE_total[:,0],'r-')
plt.plot(x_list, EE_total[:,1],'b-')
plt.plot(x_list, EE_total[:,2],'g-')
plt.show()