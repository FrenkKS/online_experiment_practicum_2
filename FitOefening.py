import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lmfit import models

df = pd.DataFrame([
    [10, 0.130],
    [30, 0.275],
    [50, 0.365],
    [70, 0.425],
    [90, 0.466],
    [110, 0.500],
    [130, 0.525],
    [150, 0.545],
    [170, 0.561],
    [190, 0.575],
    [210, 0.587],
    [230, 0.596],
    [250, 0.604],
    [270, 0.610],
    [290, 0.619],
    [310, 0.624],
], columns=['R', 'U'])

df['err_U'] = .002
df['err_R'] = .01 * df['R'] + 25e-3

df['inv_U'] = 1 / df['U']
df['inv_R'] = 1 / df['R']

df['err_inv_U'] = df['err_U'] / df['U'] * df['inv_U']
df['err_inv_R'] = df['err_R'] / df['R'] * df['inv_R']

df['I'] = df['U'] / df['R']
df['err_I'] = df['I'] * np.sqrt((df['err_U'] / df['U']) ** 2 + (df['err_R'] / df['R']) ** 2)

print(df.head())

f = lambda R, R_u, U_0: R / (R_u + R) * U_0
mod_spanning = models.Model(f, name="Spanningsdeler")

mod_linear = models.LinearModel()
fit = mod_linear.fit(df['I'], x=df['U'], weights=1/df['err_I'])
#fit = mod_linear.fit(df['inv_U'], x=df['inv_R'], weights=1/df['err_inv_U'])

#fit = mod_spanning.fit(df['U'], R=df['R'], weights=1/df['err_U'], R_u=1, U_0=1)

#df.plot.scatter('R', 'U', xerr='err_R', yerr='err_U')
#fit.plot(ylabel='U (V)', xlabel='R ($\Omega$)')
#plt.xlim(0, 350)
#plt.ylim(0, 0.7)
#plt.xlabel('R ($\Omega$)')
#plt.ylabel('U (V)')

df.plot.scatter('inv_R', 'inv_U', xerr='err_inv_R', yerr='err_inv_U')
#fit.plot(ylabel='1/U (1/V)', xlabel='1/R (1/$\Omega$)')
#plt.xlabel('1/R (1/$\Omega$)')
#plt.ylabel('1/U (1/V)')

#df.plot.scatter('U', 'I', xerr='err_U', yerr='err_I')
fit.plot(ylabel='I (A)', xlabel='U (V)')
plt.xlim(0, 0.7)
plt.ylim(0, 0.014)
plt.xlabel('U (V)')
plt.ylabel('I (A)')
plt.show()
