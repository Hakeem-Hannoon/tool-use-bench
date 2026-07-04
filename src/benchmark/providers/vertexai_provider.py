"""Vertex AI provider adapter.

Uses the same ``google-genai`` SDK as the Gemini adapter, constructed in
Vertex mode with a GCP project and location. Authentication uses Application
Default Credentials (``GOOGLE_APPLICATION_CREDENTIALS`` or ambient ADC).
"""

from __future__ import annotations

from google import genai

from ..utils.env import get_env, require_env
from .gemini_provider import GeminiProvider


class VertexAIProvider(GeminiProvider):
    name = "vertexai"

    def check_credentials(self) -> None:
        # Vertex authenticates via ADC, not an API key.
        project_env = self.cfg.project_id_env or "VERTEXAI_PROJECT_ID"
        location_env = self.cfg.location_env or "VERTEXAI_LOCATION"
        require_env(project_env)
        require_env(location_env)
        # GOOGLE_APPLICATION_CREDENTIALS is optional (ambient ADC also works),
        # so it is not required here — only used if set.

    def _make_client(self) -> genai.Client:
        project = require_env(self.cfg.project_id_env or "VERTEXAI_PROJECT_ID")
        location = get_env(self.cfg.location_env or "VERTEXAI_LOCATION") or "us-central1"
        return genai.Client(vertexai=True, project=project, location=location)
