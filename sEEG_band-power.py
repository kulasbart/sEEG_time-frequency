import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob

# directory path to data 
data_dir = ('insert path to normalized xls files') # ... insert path here

#%%

sxx_z = []
theta_p = []
alpha_p = []
beta_p = []
low_gamma_p = []

theta_z = []
alpha_z = []
beta_z = []
low_gamma_z = []

for ifile in glob.glob(data_dir):
    df = pd.read_excel(ifile)
    df = df[0:150]
    
    # set the desired frequency ranges
    theta = df[3:7].mean(axis= 0) # 4-8 Hz
    alpha = df[7:14].mean(axis= 0) # 8-13 Hz
    beta = df[14:31].mean(axis= 0) # 13-32 Hz
    low_gamma = df[31:59].mean(axis= 0) # >32Hz
    
    # z-scored power changes
    theta_z.append(theta)
    alpha_z.append(alpha)
    beta_z.append(beta)
    low_gamma_z.append(low_gamma)
    
    # 4 second baseline window starting at fourth window to avoid edge artifact in baseline
    theta = (theta - (theta[3:40].mean()))*100 
    alpha = (alpha - (alpha[3:40].mean()))*100
    beta = (beta - (beta[3:40].mean()))*100
    low_gamma = (low_gamma - (low_gamma[3:40].mean()))*100 
    
    # % change (from baseline)
    theta_p.append(theta)
    alpha_p.append(alpha)
    beta_p.append(beta)
    low_gamma_p.append(low_gamma)
    
    s = df.to_numpy()
    sxx_z.append(s)


#%% plots plots plots

t = np.linspace(0.5,14.48144531,140)
f = np.linspace(0,149,150)
t = t-5

# plotting % change from baseline in this case ... can use z-scored signal as well
theta_p_m = np.array(theta_p).mean(axis=0)
alpha_p_m = np.array(alpha_p).mean(axis=0)
beta_p_m = np.array(beta_p).mean(axis=0)
low_gamma_p_m = np.array(low_gamma_p).mean(axis=0)

#theta_z_m = np.array(theta_z).mean(axis=0)
#alpha_z_m = np.array(alpha_z).mean(axis=0)
#beta_z_m = np.array(beta_z).mean(axis=0)
#low_gamma_z_m = np.array(low_gamma_z).mean(axis=0)

sns.set_theme(style="darkgrid")
fig = plt.figure()

plt.subplot(2,2,1)
sns.lineplot(x=t, y=theta_p_m)
plt.title("theta-band response")
plt.axvline(x=0 , color='k', linestyle='--', linewidth=.5)   # ... denotes stimulus onset
plt.axvline(x=2 , color='k', linestyle='--', linewidth=.5)  
plt.xlim([-4,8])
plt.ylim([-150,150])

plt.subplot(2,2,2)
sns.lineplot(x=t, y=alpha_p_m)
plt.axvline(x=0 , color='k', linestyle='--', linewidth=.5)   
plt.axvline(x=2 , color='k', linestyle='--', linewidth=.5)  
plt.title("alpha-band response")
plt.xlim([-4,8])
plt.ylim([-150,150])

plt.subplot(2,2,3)
sns.lineplot(x=t, y=beta_p_m)
plt.title("beta-band response")
plt.axvline(x=0 , color='k', linestyle='--', linewidth=.5)   
plt.axvline(x=2 , color='k', linestyle='--', linewidth=.5)  
plt.xlim([-4,8])
plt.ylim([-150,150])

plt.subplot(2,2,4)
sns.lineplot(x=t, y=low_gamma_p_m)
plt.title("low-gamma response")
plt.axvline(x=0 , color='k', linestyle='--', linewidth=.5)   
plt.axvline(x=2 , color='k', linestyle='--', linewidth=.5)  
plt.xlim([-4,8])
plt.ylim([-150,low_gamma_p_m.max()])
plt.tight_layout()
plt.show()

#%%
# data frame with averaged power changes across subjects x power bands
df_powerbands = pd.DataFrame(columns=['theta_power','alpha_power','beta_power','low_gamma_power'])
df_powerbands['theta_power'] = theta_p_m.tolist()
df_powerbands['alpha_power'] = theta_p_m.tolist()
df_powerbands['beta_power'] = theta_p_m.tolist()
df_powerbands['low_gamma_power'] = theta_p_m.tolist()

df_powerbands.to_clipboard(excel=True,index=False)

#%%

sxx_z_m = np.sum(sxx_z[0:len(glob.glob(data_dir))-1], axis=0) / len(glob.glob(data_dir))

plt.figure(figsize=(12,8))
plt.pcolormesh(t, f, np.log10(sxx_z_m), cmap='jet',shading='gouraud')
plt.colorbar()  
plt.ylim([4,50]) # frequency range
plt.xlim([-4,6]) # time 
plt.xlabel('Time (s)')  
plt.ylabel('Frequency (Hz)')
plt.axvline(x=0 , color='k', linestyle='--', linewidth=1.2)
plt.axvline(x=2 , color='k', linestyle='--', linewidth=1.2)   
plt.clim([1,-1])  		# ... power scale (Z), should be balanced (+y,-y)
plt.show()

# ensures all files have the same dimensions
num = 0
for i in sxx_z:
    print(glob.glob(data_dir)[num])
    print(len(i))
    print(sxx_z[num].shape)
    num+=1



