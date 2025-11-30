import math

def _wrap_angle(angle):
    """Wrap angle to [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def compute_p_hit_dist(delta, limit, sigma):
    """
    Simple 'hit' probability function used by the lab exercises.

    - delta: the difference between measured and expected value (range or bearing)
    - limit: maximum allowed absolute error (beyond this probability is 0)
    - sigma: standard deviation of the measurement noise

    Behavior:
    - for angles (when `limit` is <= pi) the delta is wrapped to [-pi, pi]
    - returns 0.0 when abs(delta) > limit
    - otherwise returns the Gaussian pdf value with given sigma

    This is intentionally lightweight and compatible with the lab code.
    """
    if sigma is None or sigma <= 0:
        return 0.0

    # if limit looks like an angular limit, wrap delta
    if limit is not None and limit <= math.pi:
        delta = _wrap_angle(delta)

    if limit is not None and abs(delta) > limit:
        return 0.0

    # Gaussian PDF
    coef = 1.0 / (math.sqrt(2.0 * math.pi) * sigma)
    exponent = -0.5 * (delta / sigma) ** 2
    return coef * math.exp(exponent)
