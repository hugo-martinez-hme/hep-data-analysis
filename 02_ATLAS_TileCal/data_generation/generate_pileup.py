import numpy as np
import pandas as pd
import os


def get_pulse_shape(t, amplitude, phase):
    tau = 25.0
    return amplitude * (t / tau) * np.exp(1 - (t / tau) + phase)


def generate_event_with_pileup(main_amp, n_pileup=2):
    t = np.linspace(0, 150, 7)

    # Pulso principal
    signal = get_pulse_shape(t, main_amp, phase=0.0)

    # Creación de pulsos pile-up aleatorios
    pileup_total = np.zeros_like(t)
    for _ in range(n_pileup):
        pu_amp = np.random.uniform(5, 30)  # Pulsos secundarios más pequeños
        pu_phase = np.random.uniform(-2.0, 2.0)  # Desplazados en el tiempo
        pileup_total += get_pulse_shape(t, pu_amp, pu_phase)

    noise = np.random.normal(0, 0.8, t.shape)
    return signal + pileup_total + noise


def create_pileup_dataset(n_samples=5000):
    print(f"--- Generando {n_samples} eventos con PILE-UP ---")
    data = []
    for _ in range(n_samples):
        main_amp = np.random.uniform(10, 100)
        # Simulación de entre 1 y 3 colisiones extra por evento
        n_pu = np.random.randint(1, 4)
        samples = generate_event_with_pileup(main_amp, n_pu)
        data.append(np.append(samples, main_amp))

    df = pd.DataFrame(data, columns=[f"s{i}" for i in range(7)] + ["true_amplitude"])

    os.makedirs("../data", exist_ok=True)
    df.to_csv("../data/tilecal_pileup.csv", index=False)
    print("✅ Dataset con pile-up guardado.")


if __name__ == "__main__":
    create_pileup_dataset()
