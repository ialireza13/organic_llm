from metrics.cheap import score_m5, score_m6, score_m7
from metrics.m1_gradmax import score_m1, score_m1b
from metrics.m2_splitting import score_m2
from metrics.m3_tiny import score_m3

def score_m3neg(model, batches, opt=None):
    """Exploratory (post hoc, chosen after Stage 1): negated M3, i.e. grow where the TINY bottleneck is smallest."""
    return {k: -v for k, v in score_m3(model, batches, opt).items()}


METRICS = {
    "m5": score_m5,
    "m6": score_m6,
    "m7": score_m7,
    "m1": score_m1,
    "m1b": score_m1b,
    "m3": score_m3,
    "m2": score_m2,
    "m3neg": score_m3neg,
}
