import numpy as np

def expert_action(obs: np.ndarray) -> int:
    """
    Política ultra-segura com margens de segurança:
    - Mantém o pássaro sempre dentro dos limites seguros
    - Evita oscilação excessiva
    - Navegação suave pelos canos
    """
    y_norm, vy_norm, dist_norm, delta_gap_norm = obs
    
    # 🚨 PROTEÇÃO MÁXIMA: Margens de segurança para evitar sair da tela
    if y_norm <= 0.1:  # Muito próximo do chão (10% da tela)
        return 1
    if y_norm >= 0.9:  # Muito próximo do teto (90% da tela)
        return 0
    
    # 🛑 ANTI-OSCILAÇÃO: Evita pulos consecutivos quando subindo
    if vy_norm < -0.4:  # Subindo muito rápido
        return 0
    
    # 🎯 NAVEGAÇÃO PELOS CANOS
    if dist_norm < 0.9:  # Há cano se aproximando
        gap_center = 0.5 - delta_gap_norm  # Centro do gap normalizado
        
        # Margem de tolerância generosa
        upper_margin = 0.2
        lower_margin = 0.2
        
        # Se estou muito abaixo do centro do gap
        if y_norm > gap_center + lower_margin:
            return 1
        # Se estou muito acima do centro do gap
        elif y_norm < gap_center - upper_margin:
            return 0
    
    # 🎮 CONTROLE DE POSIÇÃO GERAL
    # Prefere ficar na parte central-baixa da tela
    if y_norm > 0.7:  # Muito alto
        return 0
    elif y_norm < 0.25:  # Muito baixo
        if vy_norm >= 0:  # Só pula se não está subindo
            return 1
    
    # 🛡️ CONTROLE DE VELOCIDADE DE QUEDA
    if vy_norm > 0.5 and y_norm > 0.3:  # Caindo rápido e não muito baixo
        return 1
    
    # Padrão: deixar a gravidade agir naturalmente
    return 0
