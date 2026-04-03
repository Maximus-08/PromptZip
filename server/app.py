"""FastAPI server for PromptZip RL Environment."""

import sys
import os

# Ensure repo root is on sys.path so `models` and `server.*` imports resolve
# when run as: uvicorn server.app:app  (from repo root or inside Docker)
_pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

from server.prompt_zip_environment import PromptZipEnvironment  # noqa: E402
from openenv.core.env_server.http_server import create_app  # noqa: E402
from models import PromptZipAction, PromptZipObservation  # noqa: E402

app = create_app(
    PromptZipEnvironment,
    PromptZipAction,
    PromptZipObservation,
    env_name="prompt_zip_env",
    max_concurrent_envs=int(os.getenv("MAX_CONCURRENT_ENVS", "4")),
)


def main() -> None:
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
