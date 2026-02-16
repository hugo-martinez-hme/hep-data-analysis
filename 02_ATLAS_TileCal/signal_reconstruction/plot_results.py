import numpy as np
import matplotlib.pyplot as plt
import joblib
import os


def generate_test_pulse(amplitude, phase, noise_level=0.8):
    t = np.linspace(0, 150, 7)
    tau = 25.0
    pulse = amplitude * (t / tau) * np.exp(1 - (t / tau) + phase)
    noise = np.random.normal(0, noise_level, t.shape)
    return t, pulse + noise


def validate_model():
    model_path = "models/tilecal_mlp_model.pkl"
    if not os.path.exists(model_path):
        print("Error: No se encuentra el modelo entrenado.")
        return

    model = joblib.load(model_path)

    test_energies = [20, 40, 60, 80]
    plt.figure(figsize=(10, 8))

    print(f"{'Real (GeV)':<12} | {'Predicho (GeV)':<15} | {'Error (GeV)':<10}")
    print("-" * 45)

    for i, true_amp in enumerate(test_energies):
        t, pulse = generate_test_pulse(true_amp, phase=0.02)

        # Predicción de la IA
        pred_amp = model.predict(pulse.reshape(1, -1))[0]
        error = abs(true_amp - pred_amp)

        print(f"{true_amp:<12.2f} | {pred_amp:<15.2f} | {error:<10.2f}")

        # Visualización
        plt.subplot(2, 2, i + 1)
        plt.plot(t, pulse, "o-", label="Muestras ADC", color="#1f77b4")
        plt.axhline(
            y=pred_amp, color="r", linestyle="--", label=f"IA: {pred_amp:.1f} GeV"
        )
        plt.title(f"Evento {i + 1}: Real {true_amp} GeV")
        plt.xlabel("Tiempo (ns)")
        plt.ylabel("Amplitud")
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/reconstruction_validation.png")
    print("\n Gráfica guardada como 'reconstruction_validation.png'")


if __name__ == "__main__":
    validate_model()
