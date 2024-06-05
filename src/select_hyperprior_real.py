from typing import Dict
from scipy.optimize import minimize
from scipy.stats import invgamma
import autograd.numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import warnings

warnings.filterwarnings("ignore")


def create_objective(q1, q99):
    def objective(x):
        # ppf: prob point function, returns value of given percentiles
        # x are the objective to min/max
        aq1, aq99 = invgamma.ppf([0.01, 0.99], x[0], scale=x[1])

        return (aq1 - q1) ** 2 + (aq99 - q99) ** 2

    return objective


def select_hyperprior(q1, q99, init1=1.0, init2=1.0):
    # q1s = [0.01, 0.1, 1.0, 10.0, 100]
    res = minimize(
        create_objective(q1, q99), [init1, init2], bounds=[(0, np.inf), (0, np.inf)]
    )
    return res["x"]
    # x = np.linspace(
    #     invgamma.ppf(0.01, a=res["x"][0], scale=res["x"][1]),
    #     invgamma.ppf(0.99, a=res["x"][0], scale=res["x"][1]),
    #     1000,
    # )
    # plt.plot(
    #     x,
    #     invgamma.pdf(x, res["x"][0], scale=res["x"][1]),
    #     "r-",
    #     lw=5,
    #     alpha=0.6,
    #     label="invgamma pdf",
    # )
    # plt.savefig("fig.png")


cancer_categories = {
    "skin": "Skin-Melanoma",
    "ovary": "Ovary-AdenoCA",
    "breast": "Breast",
    "liver": "Liver-HCC",
    "lung": "Lung-SCC",
    "stomach": "Stomach-AdenoCA",
}
result = dict()

for cancer in cancer_categories:
    data_dir = "../WGS_PCAWG.96.ready/{}.tsv".format(cancer_categories[cancer])
    data = pd.read_csv(os.path.join(data_dir), sep="\t")
    X = data.values[:, 1:]
    countsPerSample = np.sum(X, axis=0)
    meanCountsPerSample = np.mean(countsPerSample, axis=-1)
    # q1s = [0.01, 10, 100]
    q1 = 10
    q99 = meanCountsPerSample / 2.0
    result["{}".format(cancer)] = select_hyperprior(q1, q99)

for key in result:
    print(key, end="\t")

for i in range(2):
    print()
    for key in result:
        print(result[key][i], end="\t")