import re


def validate_physics_response(answer: str) -> str:
    """Valida que la masa del Z esté en el rango físico."""
    mass_match = re.search(r"M\s*=\s*(\d+\.?\d*)", answer)
    if mass_match:
        m_value = float(mass_match.group(1))
        # Filtro: si menciona Z pero la masa es absurda
        if ("z" in answer.lower() or "bosón" in answer.lower()) and (
            m_value < 70 or m_value > 110
        ):
            return f"⚠️ ERROR FÍSICO: Se reportó un Bosón Z con M={m_value} GeV (Inconsistente)."
    return answer
