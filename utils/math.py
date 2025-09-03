import numpy as np


def lemniscate_trajectory(t, a=0.2, offset:np.ndarray=np.array([0.5,0,0.6])):
    """Given t, returns the (x, y, z) position on a lemniscate trajectory.
    ref: https://en.wikipedia.org/wiki/Lemniscate_of_Bernoulli
    """
    assert offset.shape == (3,)
    s = np.sin(t)
    c = np.cos(t)
    
    x = a * np.sin(t) * np.cos(t) / (1 + np.sin(t)**2) + offset[0]
    y = a * np.cos(t) / (1 + np.sin(t)**2) + offset[1]
    z = offset[2]
    
    vx = a * (c**2 - s**2 - s**4 - (c**2) * (s**2)) / ((1 + s**2)**2)
    vy = a * ( -s - s**3 - 2*s*c**2 ) / ( (1 + s**2)**2 )
    vz = 0
    
    ax = 0
    ay = 0
    az = 0

    return np.array([x, y, z]), np.array([vx, vy, vz]), np.array([ax, ay, az])


if __name__ == "__main__":
    pass
