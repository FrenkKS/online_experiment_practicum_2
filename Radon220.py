import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lmfit import models
import lmfit

df = pd.read_csv('Radon220V2.csv')
df['N1_err'] = np.sqrt(df['N1'])
df['N2_err'] = np.sqrt(df['N2'])
df['N3_err'] = np.sqrt(df['N3'])

sel = df.query('(t >= 63)')

f = lambda t, N_0, l, Bv: N_0 * np.exp(-(l*t)) + Bv
mod_halfwaarde = models.Model(f, name='intensiteit')

fit = mod_halfwaarde.fit(sel['N1'], t=sel['t'], weights=1/sel['N1_err'], N_0 = 400, l = 0.01, Bv = 20)

sel.plot.scatter('t', 'N1', yerr='N1_err')
fit.plot(ylabel='N', xlabel='t (s)')
sel.plot.scatter('t', 'N2', yerr='N2_err')
sel.plot.scatter('t', 'N3', yerr='N3_err')
plt.show()

print(lmfit.report_fit(fit))