import pandas as pd
import numpy as np
import os


def create_mixed_dataset():
    clean_path = "../data/tilecal_pulses.csv"
    pileup_path = "../data/tilecal_pileup.csv"

    if not os.path.exists(clean_path) or not os.path.exists(pileup_path):
        print(
            "Error: Faltan archivos de datos. Genera primero los de pile-up y los limpios."
        )
        return

    df_clean = pd.read_csv(clean_path)
    df_pileup = pd.read_csv(pileup_path)

    # Mezclamos ambos
    df_mixed = pd.concat([df_clean, df_pileup]).sample(frac=1).reset_index(drop=True)

    df_mixed.to_csv("../data/tilecal_hardened_data.csv", index=False)
    print(f"Dataset mixto creado con {len(df_mixed)} ejemplos.")


if __name__ == "__main__":
    create_mixed_dataset()
