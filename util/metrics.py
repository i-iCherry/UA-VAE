"""
Contain some metrics
"""
import numpy as np
# from lifelines.utils import concordance_index
# from pysurvival.utils._metrics import _concordance_index
from sksurv.metrics import concordance_index_censored
from sksurv.metrics import integrated_brier_score


def c_index(true_T, true_E, pred_risk, include_ties=True):
    """
    Calculate c-index for survival prediction downstream task
    """
    # Ordering true_T, true_E and pred_score in descending order according to true_T
    order = np.argsort(-true_T)

    true_T = true_T[order]
    true_E = true_E[order]
    pred_risk = pred_risk[order]

    # Calculating the c-index
    # result = concordance_index(true_T, -pred_risk, true_E)
    # result = _concordance_index(pred_risk, true_T, true_E, include_ties)[0]
    result = concordance_index_censored(true_E.astype(bool), true_T, pred_risk)[0]

    return result


