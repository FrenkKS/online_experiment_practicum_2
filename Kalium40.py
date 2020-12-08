import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lmfit import models
import lmfit

df_1 = pd.DataFrame([
    [7, 1399, 187],
    [9, 1455, 159],
    [10, 1421, 212],
    [11, 1580, 154],
    [13, 1418, 177],
    [15, 1589, 183],
], columns=['GM', 'counts', 'achtergrond'])

df_1['counts_err'] = np.sqrt(df_1['counts'])
df_1['achtergrond_err'] = np.sqrt(df_1['achtergrond'])

df_1['strontium'] = df_1['counts'] - df_1['achtergrond']
df_1['strontium_err'] = np.sqrt((df_1['counts_err'])**2 + (df_1['achtergrond_err'])**2)

def geometrisch(r, R):
    return (r**2)/(4*R**2)

def geometrisch_err(r, R, r_err, R_err):
    return np.sqrt(((r*r_err)/(2*R**2))**2 + ((-(r**2)*R_err)/(2*R**3))**2)

e_g = geometrisch(0.015, 0.15)
e_g_err = geometrisch_err(0.015, 0.15, 0.001, 0.001)

strontium_totaal = 960 * 1400
strontium_totaal_err = 960

df_1['efficientie'] = df_1['strontium']/strontium_totaal
df_1['efficientie_err'] = np.sqrt((df_1['strontium_err']/strontium_totaal)**2 + (-(df_1['strontium']*strontium_totaal_err)/strontium_totaal**2)**2)

df_1['intrinsiek'] = df_1['efficientie']/e_g
df_1['intrinsiek_err'] = np.sqrt((df_1['efficientie_err']/e_g)**2 + (-(df_1['efficientie']*e_g_err)/e_g**2)**2)

mod_linear = models.ConstantModel()
fit = mod_linear.fit(df_1['strontium'], x=df_1['GM'], weights=1/df_1['strontium_err'])

# print(lmfit.report_fit(fit))

# print(fit.redchi)
## referentie: https://stackoverflow.com/questions/43381833/lmfit-extract-fit-statistics-parameters-after-fitting
# print(fit.params['c'].value)

intrinsiek = df_1['intrinsiek'].tolist()
intrinsiek_err = df_1['intrinsiek_err'].tolist()

intrinsiek_err_sqrd = []

for error in intrinsiek_err:
    intrinsiek_err_sqrd.append(float(error)**2)

intrinsiek_7 = sum(intrinsiek)/6
intrinsiek_7_err = np.sqrt(sum(intrinsiek_err_sqrd))/6

# intrinsiek_7 = 0.4
# intrinsiek_7_err = 0.02

print(f"De intrinsieke efficiëntie van de 7e GM-buis is {intrinsiek_7:.2f} +/- {intrinsiek_7_err:.2f}")

molmassa_kacarb = 2*39.0983 + 12.011 + 3*15.9994
molmassa_kacarb_err = np.sqrt((2*0.0001)**2 + 0.001**2 + (3*0.0001)**2)

df = pd.DataFrame([
    [0.171, 433],
    [0.274, 514],
    [0.378, 651],
    [0.632, 792],
    [1.045, 893],
    [1.353, 939],
], columns=['hoeveelheid_kacarb', 'counts'])

df['hoeveelheid_kacarb_err'] = 0.004
df['counts_err'] = np.sqrt(df['counts'])

df['mol_kacarb'] = df['hoeveelheid_kacarb']/molmassa_kacarb
df['mol_kacarb_err'] = np.sqrt((df['hoeveelheid_kacarb_err']/molmassa_kacarb)**2 + ((df['hoeveelheid_kacarb']*molmassa_kacarb_err)/molmassa_kacarb**2)**2)

df['mol_kalium'] = 2 * df['mol_kacarb']
df['mol_kalium_err'] = 2 * df['mol_kacarb_err']

avogadro = 6.02214076*(10**23)

df['aantal_kalium_deeltjes'] = avogadro * df['mol_kalium']
df['aantal_kalium_deeltjes_err'] = avogadro * df['mol_kalium_err']

fractie_kalium_40 = 1.17e-4
fractie_kalium_40_err = .01e-4

df['aantal_kalium_40_deeltjes'] = fractie_kalium_40 * df['aantal_kalium_deeltjes']
df['aantal_kalium_40_deeltjes_err'] = np.sqrt((df['aantal_kalium_deeltjes'] * fractie_kalium_40_err)**2 + (fractie_kalium_40 * df['aantal_kalium_deeltjes_err'])**2)

df['gemeten'] = df['counts'] - 159
df['gemeten_err'] = np.sqrt((df['counts_err'])**2 + 159)

geometrisch_kalium = 0.4
geometrisch_kalium_err = 0.07

efficientie = geometrisch_kalium * intrinsiek_7
efficientie_err = np.sqrt((geometrisch_kalium*intrinsiek_7_err)**2 + (intrinsiek_7*geometrisch_kalium_err)**2)

df['t_half'] = (df['aantal_kalium_40_deeltjes'] * np.log(2) * efficientie * 600) / df['gemeten']
df['t_half_err'] = df['t_half'] * np.sqrt((df['aantal_kalium_deeltjes_err'] / df['aantal_kalium_deeltjes'])**2 + (efficientie_err / efficientie)**2 + (1 / 600)**2 + (df['gemeten_err'] / df['gemeten'])**2)

df['t_half_jaar'] = df['t_half'] / (365.25 * 24 * 60 * 60)
df['t_half_jaar_err'] = df['t_half_err'] / (365.25 * 24 * 60 * 60)

# print(df[['aantal_kalium_deeltjes', 'aantal_kalium_40_deeltjes', 'aantal_kalium_40_deeltjes_err', 't_half', 't_half_err', 't_half_jaar', 't_half_jaar_err']])

f = lambda x, t_half, mu: t_half * np.exp(mu*x)
exponential = models.Model(f, name='halfwaardetijd')

fit_2 = exponential.fit(df['t_half_jaar'], x=df['hoeveelheid_kacarb'], weights=1/df['t_half_jaar_err'], t_half=1.25e9, mu=0.8)

fit_2.plot(ylabel='t_half (y)', xlabel='kaliumcarbonaat (g)')
plt.show()

print(lmfit.report_fit(fit_2))

t_half = fit_2.params['t_half'].value
t_half_err = fit_2.params['t_half'].stderr

print(f'De halfwaardetijd is {t_half:.3e} jaar +/- {t_half_err:.3e} jaar')

