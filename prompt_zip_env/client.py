"""Client for the PromptZip RL Environment."""

from typing import Any, Dict, Optional
from openenv.core.env_client import EnvClient

from .models import PromptZipAction, PromptZipObservation


class PromptZipEnv(EnvClient):
    """Typed client for interacting with a running PromptZip environment server."""

    action_type = PromptZipAction
    observation_type = PromptZipObservation

    @classmethod
    def from_docker_image(
        cls,
        image_name: str = "prompt_zip_env:latest",
        port: int = 8000,
        env_vars: Optional[Dict[str, str]] = None,
        timeout: int = 30,
        **kwargs: Any,
    ) -> "PromptZipEnv":
        """
        Spin up a local Docker container and return a connected client.
        Test manually before push:
            env = PromptZipEnv.from_docker_image(env_vars={"GROQ_API_KEY": "..."})
            with env.sync() as client:
                obs = client.reset()
        """
        from openenv.core.providers.local_docker import LocalDockerProvider  # type: ignore

        provider = LocalDockerProvider(
            image=image_name,
            port=port,
            env_vars=env_vars or {},
            timeout=timeout,
        )
        return cls(base_url=f"http://localhost:{port}", **kwargs)
