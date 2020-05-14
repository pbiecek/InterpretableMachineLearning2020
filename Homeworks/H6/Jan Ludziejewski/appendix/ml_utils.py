import numpy as np
from utils import dict_with
from tqdm import tqdm, trange
import math


def local_search(params, score_function, limits=None, upper_rel=1.1, lower_rel=0.8, iterations=100):
    params = params.copy()
    limits = {} if limits is None else limits
    last_score = -np.inf
    for iteration in trange(iterations):
        no_change = True
        for key in list(params.keys()):
            value = params[key]
            print("Checking {} in area of {}...".format(key, value), end='')
            proposals = [value * lower_rel, value, value * upper_rel] if type(value) is not int else\
                [math.floor(value * lower_rel), value, math.ceil(value * upper_rel + 1e-5)]
            lmin, lmax = limits.get(key, (-np.inf, np.inf))
            proposals[0], proposals[2] = max(proposals[0], lmin), min(proposals[2], lmax)
            scores = [score_function(dict_with(params, key, x)) for x in proposals]
            best: int = np.argmax(scores)
            params[key] = proposals[best]
            no_change = no_change and best == 1
            last_score = scores[best]
            status = "".join(["/" if scores[x + 0] < scores[x + 1] else "=" if scores[x + 0] == scores[x + 1] else "\\"
                              for x in [0, 1]])
            print("({}) chosen = {} with score {}".format(status, proposals[best], scores[best]))

        print("{}: score = {}\n {}".format(iteration, last_score, params))
        if no_change:
            print("Early stopping after encountering local minimum")
            break
    return params



