import torch
import numpy as np
import pytest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from signal_reconstruction.train_pinn_reconstructor import (
    TileCalPINN,
    pulse_shape_physics,
)


def test_model_output_shape():
    """Verifica que el modelo devuelva la forma correcta (Batch, 1)"""
    model = TileCalPINN()
    dummy_input = torch.randn(10, 7, 1)  # Batch de 10 eventos
    output = model(dummy_input)
    assert output.shape == (10, 1)


def test_physics_consistency():
    """Verifica que la función física genere valores positivos para amplitudes positivas"""
    t = torch.tensor([25.0, 50.0])
    amp = torch.tensor([50.0])
    pulse = pulse_shape_physics(t, amp)
    assert torch.all(pulse >= 0)
    # El pico teórico en t=tau (25) debería ser igual a la amplitud (50)
    assert torch.isclose(pulse[0], amp)


def test_model_loading():
    """Verifica que el modelo guardado se pueda cargar correctamente"""
    model_path = "signal_reconstruction/models/tilecal_pinn_model.pth"
    if os.path.exists(model_path):
        model = TileCalPINN()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        assert True
    else:
        pytest.skip("Modelo no encontrado para cargar")
