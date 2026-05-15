from pathlib import Path
import SMEFT19
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from flavio.statistics.functions import pull
import warnings
from flavio.statistics.functions import pull, delta_chi2, pvalue
from SMEFT19 import likelihood_global
from SMEFT19.scenarios import rotBIII
warnings.simplefilter("ignore")

DATA_DIR = Path(__file__).parent.parent / 'data'
DATA_DIR.mkdir(exist_ok=True)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    SMEFT19.SMEFTglob.gl.make_measurement()

def fit_II(x):
    try:
        return -SMEFT19.likelihood_global(x, SMEFT19.scenarios.rotBIII)
    except:
        return -2000

bf_II, v_II, d_II, L_II = SMEFT19.ellipse.minimum(fit_II, [-0.13, -0.13, -0.76])

SMEFT19.ellipse.save(bf_II, v_II, d_II, L_II, str(DATA_DIR / 'rotBIII.yaml'),
                     name='Mass Rotation fit, Scenario III', fit='rotBIII')
L0 = fit_II(np.zeros(3))

with open(DATA_DIR / 'results_03.txt', 'wt') as f:
    f.write(f'Best fit point: {str(bf_II)}\n')
    f.write(f'Delta Chi2: {2*L_II}\n')
    f.write(f'Pull: {pull(2*abs(L_II-L0), 3)}')

d_ell = SMEFT19.ellipse.load(str(DATA_DIR / 'rotBIII.yaml'))
bf1 = d_ell['bf']


def lh_C1C3(num: int) -> float:
    xmin = -0.3
    xmax = 0.0
    ymin = -0.3
    ymax = 0.0
    xmargin = 0.02*(xmax-xmin)
    ymargin = 0.02*(ymax-ymin)
    ix = num % 50
    iy = num // 50
    x = (xmin-xmargin) + ix/50 * ((xmax+xmargin) - (xmin-xmargin))
    y = (ymin-ymargin) + iy/50 * ((ymax+ymargin) - (ymin-ymargin))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            lg = likelihood_global([x, y, bf1[2]], rotBIII)
        except:
            return -100
        return max(lg, -100)

def lh_C1beta(num: int) -> float:
    xmin = -0.3
    xmax = 0.0
    ymin = 0.0
    ymax = 3.2
    xmargin = 0.02*(xmax-xmin)
    ymargin = 0.02*(ymax-ymin)
    ix = num % 50
    iy = num // 50
    x = (xmin-xmargin) + ix/50 * ((xmax+xmargin) - (xmin-xmargin))
    y = (ymin-ymargin) + iy/50 * ((ymax+ymargin) - (ymin-ymargin))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            lg = likelihood_global([x, bf1[1], y], rotBIII)
        except:
            return -100
        return max(lg, -100)
    
def lh_C3beta(num: int) -> float:
    xmin = -0.3
    xmax = 0.0
    ymin = 0.0
    ymax = 3.2
    xmargin = 0.02*(xmax-xmin)
    ymargin = 0.02*(ymax-ymin)
    ix = num % 50
    iy = num // 50
    x = (xmin-xmargin) + ix/50 * ((xmax+xmargin) - (xmin-xmargin))
    y = (ymin-ymargin) + iy/50 * ((ymax+ymargin) - (ymin-ymargin))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            lg = likelihood_global([bf1[0], x, y], rotBIII)
        except:
            return -100
        return max(lg, -100)

start = 0

with open(DATA_DIR / 'likelihood_rotBIII_C1C3.dat', 'at') as f:
    for i in range(start, 50*50):
        lg = lh_C1C3(i)
        if i%50 == 49:
            sep = '\n'
        else:
            sep = '\t'
        f.write(f'{lg}{sep}')
        f.flush()
        print(i)
        
start = 0

with open(DATA_DIR / 'likelihood_rotBIII_C1beta.dat', 'at') as f:
    for i in range(start, 50*50):
        lg = lh_C1beta(i)
        if i%50 == 49:
            sep = '\n'
        else:
            sep = '\t'
        f.write(f'{lg}{sep}')
        f.flush()
        print(i)
        
start = 0

with open(DATA_DIR / 'likelihood_rotBIII_C3beta.dat', 'at') as f:
    for i in range(start, 50*50):
        lg = lh_C3beta(i)
        if i%50 == 49:
            sep = '\n'
        else:
            sep = '\t'
        f.write(f'{lg}{sep}')
        f.flush()
        print(i)