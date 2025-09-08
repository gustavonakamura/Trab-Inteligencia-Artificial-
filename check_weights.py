#!/usr/bin/env python3
import numpy as np
import os

print("🔍 Verificando formato dos arquivos de pesos...")

files_to_check = [
    'weights.npy',
    'runs/best_weights.npy',
    'runs/run_1_weights.npy'
]

for file in files_to_check:
    if os.path.exists(file):
        print(f"\n📁 {file}:")
        try:
            data = np.load(file, allow_pickle=True)
            print(f"   Tipo: {type(data)}")
            print(f"   Shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")
            print(f"   Conteúdo: {data}")
        except Exception as e:
            print(f"   Erro: {e}")
    else:
        print(f"\n❌ {file}: Não encontrado")
