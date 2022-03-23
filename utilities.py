import numpy as np

def argmax(x):
    q_max = float('-inf')
    q_max_index = []
    for i, q in enumerate(x.squeeze()):
        if q > q_max:
            q_max = q
            q_max_index = [i]
        elif q == q_max:
            q_max_index.append(i)
    
    return np.random.choice(q_max_index)
            