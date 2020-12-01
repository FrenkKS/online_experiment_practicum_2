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

print(f"De intrinsieke efficiëntie van de 7e GM-buis is {intrinsiek_7:.2f} +/- {intrinsiek_7_err:.2f}")

