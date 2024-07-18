"""Checks related to the .env file in the repository."""

from pathlib import Path

DESIRED_ENVIRONMENT_VARIABLES = dict(
    GIT_NAME="Enter your full name, to be shown in Git commits:\n> ",
    GIT_EMAIL="Enter your email, as registered on your Github account:\n> ",
    PYPI_API_TOKEN="Enter your PyPI API token, or leave empty if you do not want "
    "to use it:\n> ",
)


def fix_dot_env_file():
    """Ensures that the .env file exists and contains all desired variables."""
    env_file_path = Path(".env")
    env_file_path.touch(exist_ok=True)

    # Extract all the lines and environment variables present in the .env file
    env_file_lines = env_file_path.read_text().splitlines(keepends=False)
    env_vars = [line.split("=")[0] for line in env_file_lines]

    env_vars_missing = [
        env_var
        for env_var in DESIRED_ENVIRONMENT_VARIABLES.keys()
        if env_var not in env_vars
    ]

    # Create all the missing environment variables
    with env_file_path.open("a") as f:
        for env_var in env_vars_missing:
            value = ""
            if value == "":
                value = input(DESIRED_ENVIRONMENT_VARIABLES[env_var])
            f.write(f'{env_var}="{value}"\n')


if __name__ == "__main__":
    fix_dot_env_file()
