import pyedflib
from scipy.fftpack import fft
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import spectrogram
import pandas as pd
import glob

# file directory
files = glob.glob('')
files

count = 0
sxx_z = []
theta_p = []
alpha_p = []
beta_p = []
low_gamma_p = []

while count < len(files):
    df = pd.read_excel(files[count])
    df = df[0:150]
    theta = df[4:9].to_numpy().mean(axis= 0)
    alpha = df[8:14].to_numpy().mean(axis= 0)
    beta = df[13:33].to_numpy().mean(axis= 0)
    low_gamma = df[32:60].to_numpy().mean(axis= 0)
    
    theta = (theta - (theta[3:36].mean()))*100
    alpha = (alpha - (alpha[3:36].mean()))*100
    beta = (beta - (beta[3:36].mean()))*100
    low_gamma = (low_gamma - (low_gamma[3:36].mean()))*100 
        
    theta_p.append(theta)
    alpha_p.append(alpha)
    beta_p.append(beta)
    low_gamma_p.append(low_gamma)
    
    s = df.to_numpy()
    sxx_z.append(s)
    count += 1

t = np.linspace(0.5,14.48144531,140)
f = np.linspace(0,149,150)
t = t-5

#%% plots plots plots

theta_p_m = np.array(theta_p).mean(axis=0)
alpha_p_m = np.array(alpha_p).mean(axis=0)
beta_p_m = np.array(beta_p).mean(axis=0)
low_gamma_p_m = np.array(low_gamma_p).mean(axis=0)

sns.set_theme(style="darkgrid")
sns.lineplot(x=t, y=theta_p_m)
plt.title("theta-band response")
plt.xlim([-4,10])
plt.ylim([-150,150])
plt.show()

sns.lineplot(x=t, y=alpha_p_m)
plt.title("alpha-band response")
plt.xlim([-4,10])
plt.ylim([-150,150])
plt.show()

sns.lineplot(x=t, y=beta_p_m)
plt.title("beta-band response")
plt.xlim([-4,10])
plt.ylim([-150,150])
plt.show()

sns.lineplot(x=t, y=low_gamma_p_m)
plt.title("low-gamma response")
plt.xlim([-4,10])
plt.ylim([-150,250])
plt.show()

#%%

sxx_z_m = np.sum(sxx_z[0:count-1], axis=0) / count

plt.figure(figsize=(12,8))
plt.pcolormesh(t, f, np.log10(sxx_z_m), cmap='jet',shading='gouraud')
plt.colorbar() 
plt.ylim([4,50])
#plt.xlim([-4,6])
plt.xlabel('Time (s)')  
plt.ylabel('Frequency (Hz)')
plt.axvline(x=0 , color='k', linestyle='--', linewidth=1.2)
plt.axvline(x=2 , color='k', linestyle='--', linewidth=1.2)   
plt.clim([1,-1])  		# ... power scale (Z), should be balanced (+y,-y)
plt.show()

# ensure all files have the same dimensions
num = 0
for i in sxx_z:
    print(files[num])
    print(len(i))
    print(sxx_z[num].shape)
    num+=1