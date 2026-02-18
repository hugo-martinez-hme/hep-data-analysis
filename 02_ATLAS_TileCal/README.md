# 02_ATLAS_TileCal: Advanced Signal Reconstruction

This module focuses on energy reconstruction for the **ATLAS Tile Calorimeter (TileCal)** using various Deep Learning architectures. The core objective is to accurately estimate the true amplitude of a particle's energy deposit from a sequence of digitized samples, particularly in high-luminosity environments with significant **pile-up**.

## 🧬 Physics Context
The ATLAS TileCal measures energy by sampling the electronic signal produced by scintillation light. A typical signal consists of **7 samples** spaced every 25 ns. Accurately reconstructing the peak amplitude is vital for determining the energy of jets and missing transverse momentum (MET).

The main challenge addressed here is **pile-up**: extra overlapping pulses from multiple proton-proton collisions that distort the main signal.

## 🚀 Key Features

- **Synthetic Data Generation**: Realistic pulse simulation based on the theoretical TileCal shape, including Gaussian noise and out-of-time pile-up.
- **Physics-Informed Neural Networks (PINNs)**: Custom architecture where the loss function incorporates the theoretical pulse shape to ensure the model respects detector physics.
- **Recurrent Neural Networks (RNN/LSTM)**: Models that treat the 7 samples as a time-series to capture temporal evolution.
- **Pile-up Hardening**: Training strategies using mixed datasets to improve model resilience in high-occupancy conditions.

## 📂 Project Structure

* **`data/`**: Storage for generated synthetic datasets (CSV).
* **`data_generation/`**: Scripts to simulate physics environments.
* **`signal_reconstruction/`**: Core training and evaluation logic.
    * **`models/`**: Production-ready model artifacts and normalization parameters.
    * **`results/`**: Graphical validation of reconstruction performance.
    * `train_pinn_reconstructor.py`: Physics-Informed learning implementation.
    * `train_rnn_reconstruction.py`: LSTM-based sequence analysis.
    * `plot_results.py`: Visualization tool for comparing predictions vs. truth.
* **`tests/`**: Unit tests for model consistency and physics validity.

## 📦 Model Artifacts (`/models`)

The directory stores trained weights and pre-processing parameters:
- **`tilecal_pinn_model.pth`**: Weights for the Physics-Informed Neural Network (PyTorch).
- **`tilecal_rnn_model.pth`**: Weights for the LSTM Reconstructor (PyTorch).
- **`tilecal_mlp_model.pkl`**: Scikit-Learn MLP Regressor for baseline comparison.
- **`tilecal_hardened_model.pkl`**: Model trained specifically for pile-up resilience.
- **`scaling_params.pkl`**: Mean and standard deviation used for feature scaling.

## 📊 Results and Validation (`/results`)

Visual evidence of the model's accuracy is stored here.
