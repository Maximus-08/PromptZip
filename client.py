"""Client for the PromptZip RL Environment."""

from typing import Any, Optional
from openenv.core.env_client import EnvClient

try:
    from .models import PromptZipAction, PromptZipObservation  # package import
except ImportError:
    from models import PromptZipAction, PromptZipObservation  # flat / Docker / test


class PromptZipEnv(EnvClient):
    """Typed client for interacting with a running PromptZip environment server."""

    action_type = PromptZipAction
    observation_type = PromptZipObservation

    @classmethod
    def from_docker_image(
        cls,
        image_name: str = "prompt_zip_env:latest",
        port: int = 8000,
        **kwargs: Any,
    ) -> "PromptZipEnv":
        """
        Return a client connected to a locally-running Docker container.
        The caller is responsible for starting the container first:

            docker run -p 8000:8000 -e GROQ_API_KEY=<key> prompt_zip_env:latest

        Then connect:

            with PromptZipEnv.from_docker_image().sync() as client:
                obs = client.reset()
        """
        return cls(base_url=f"http://localhost:{port}", **kwargs)
