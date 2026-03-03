import argparse

import uvicorn

from speaches.dependencies import get_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Speaches server")
    parser.add_argument(
        "--idle-offload-seconds",
        type=int,
        default=None,
        help="STT idle offload timeout in seconds. Use -1 to disable offloading.",
    )
    args = parser.parse_args()

    get_config.cache_clear()
    config = get_config()
    config.idle_offload_cli_seconds = args.idle_offload_seconds

    uvicorn.run("speaches.main:create_app", factory=True, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
