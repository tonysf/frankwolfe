"""Presentation names for stable recorded method IDs; no algorithm changes."""
METHOD_NAMES = {
    "smoothing-fw": "Smoothed row FW + momentum (sketch gradient)",
    "smoothing-fw-exact": "Smoothed row FW + momentum (exact gradient)",
    "full-kl": "Transport KL-proximal gradient (exact gradient)",
    "stochastic-kl": "Transport KL-proximal gradient (sketch gradient)",
    "stochastic-kl-momentum": "Transport KL-proximal gradient + momentum (sketch gradient)",
    "native-sampled-gw": "Native Sampled GW (POT)",
    "fw": "Transport FW + line search (exact gradient)",
    "smoothing-row-projection": "Smoothed row projection + momentum (sketch gradient)",
    "pg": "Transport projected gradient (exact gradient)",
}

def method_name(identifier):
    return METHOD_NAMES.get(identifier, identifier)
