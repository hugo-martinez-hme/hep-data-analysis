from typing import Dict, Any, Optional
from archi.pipelines.agents.base_react import BaseReActAgent
from src.utils.logging import get_logger

logger = get_logger(__name__)


class CMSPhysicsAgent(BaseReActAgent):
    """Agente especializado en análisis de física de CMS heredando de Archi Core."""

    def __init__(self, config: Dict[str, Any], *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.agent_prompt = (
            "Eres un analista de datos del CMS. Tu única fuente de verdad es la herramienta 'cms_data_analyzer'. "
            "NO INVENTES ESTADÍSTICAS. Si no recibes una tabla de datos de la herramienta, "
            "di que no puedes realizar el análisis. "
            "Tus respuestas DEBEN incluir la tabla de datos recibida."
        )
        self.analysis_threshold = 40.0
        logger.info("Agente CMS Physics Analyst inicializado con éxito.")

    def _build_static_middleware(self):
        """Aquí inyectaremos tu 'capa de escepticismo' más adelante."""
        return super()._build_static_middleware()
