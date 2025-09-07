import numpy as np

def expert_action(obs: np.ndarray) -> int:
    """
    Heurística:
      - Se estou abaixo do centro do gap (delta_gap_norm > 0) e o próximo cano está próximo (dist_norm < 0.25),
        avalio a velocidade: se estou caindo (vy_norm > -0.1), pulo.
      - Se estou muito abaixo do gap (delta_gap_norm > 0.12), pulo independente da distância.
      - Caso contrário, não pulo.
    """
    y_norm, vy_norm, dist_norm, delta_gap_norm = obs
    if delta_gap_norm > 0.12:
        return 1
    if dist_norm < 0.25 and delta_gap_norm > 0.0 and vy_norm > -0.1:
        return 1
    return 0
