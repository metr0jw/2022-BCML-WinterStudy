# code from StackExchange Statistics
# https://stats.stackexchange.com/questions/539496/how-to-create-hinge-loss-function-in-python-from-scratch

import numpy as np
from sklearn.metrics import hinge_loss


def hl_func(actual, predicted):
    new_predicted = np.array([-1 if i == 0 else i for i in predicted])
    new_actual = np.array([-1 if i == 0 else i for i in actual])

    # calculating hinge loss
    hinge_loss = np.mean([max(0, 1-x*y) for x, y in zip(new_actual, new_predicted)])
    return hinge_loss


# case 1
actual = np.array([1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1])
predicted = np.array([0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1])
new_predicted = np.array([-1 if i == 0 else i for i in predicted])
print("# case 1")
print("scikit-learn hinge loss output: " + str(hinge_loss(actual, new_predicted)))
print("implemented hinge loss output: " + str(hl_func(actual, predicted)) + '\n')

# case 2
actual = np.array([1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1])
predicted = np.array([0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1])
new_predicted = np.array([-1 if i == 0 else i for i in predicted])
print("# case 2")
print("scikit-learn hinge loss output: " + str(hinge_loss(actual, new_predicted)))
print("implemented hinge loss output: " + str(hl_func(actual, predicted)))
