from smt.applications import MFK
import numpy as np
from matplotlib import pyplot as plt
from smt.applications.mfk import NestedLHS
from smt.surrogate_models import KRG


def LF_function(x):
    return 0.5 * ((x * 6 - 2) ** 2) * np.sin((x * 6 - 2) * 2) + (x - 0.5) * 10.0 - 5


def HF_function(x):
    return 0.5 * ((x * 6 - 2) ** 2) * np.sin((x * 6 - 2) * 2)


ndim = 1
nlvl = 2
ndoe_HF = 4

xlimits = np.array([[0.0, 1.0]])
xdoes = NestedLHS(nlevel=nlvl, xlimits=xlimits, random_state=2)
Xt_c, Xt_e = xdoes(ndoe_HF)
ndoe_LF = np.shape(Xt_c)[0]

yt_e = HF_function(Xt_e)
yt_c = LF_function(Xt_c)

sm = MFK(theta0=Xt_e.shape[1] * [1.0])

sm.set_training_values(Xt_c, yt_c, name=0)
sm.set_training_values(Xt_e, yt_e)

sm.train

# predictvalues
x = np.linspace(0, 1, 101, endpoint=True).reshape(-1, 1)
y = sm.predict_values(x)
