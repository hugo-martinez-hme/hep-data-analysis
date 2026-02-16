import numpy as np
import pandas as pd
import os


def generate_pulse(amplitude, phase, noise_level=0.5):
    t = np.linspace(0, 150, 7)
    tau = 25.0

    pulse = amplitude * (t / tau) * np.exp(1 - (t / tau) + phase)
    noise = np.random.normal(0, noise_level, t.shape)
    return pulse + noise


def create_dataset(n_samples=10000):
    print(f"--- Generando {n_samples} pulsos sintéticos de ATLAS ---")
    data = []
    for _ in range(n_samples):
        amp = np.random.uniform(10, 100)
        phase = np.random.uniform(-0.1, 0.1)
        pulse = generate_pulse(amp, phase)
        data.append(np.append(pulse, amp))

    columns = [f"s{i}" for i in range(7)] + ["true_amplitude"]
    df = pd.DataFrame(data, columns=columns)

    output_path = "../data/tilecal_pulses.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Dataset guardado en: {output_path}")


if __name__ == "__main__":
    create_dataset()
