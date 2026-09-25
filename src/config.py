"""Application configuration module.

Contains project settings loaded from environment variables and .env configuration files.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings gathered from env / .env.

    Attributes:
        app_name: The name of the web application.
        version: Current semantic version (SemVer).
        log_level: Logging level (DEBUG, INFO, WARNING, etc.).
        debug: Debug mode flag.
        environment: Deployment environment (development, production).
        database_url: PostgreSQL connection string in asyncpg format.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "agrodiagnosis"
    version: str = "0.1.0"
    log_level: str = "INFO"
    debug: bool = False
    environment: str = "development"
    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/agrodiagnosis",
        description="PostgreSQL connection string (asyncpg format).",
    )


def get_settings() -> Settings:
    """Returns an instance of the application settings.

    Used for Dependency Injection in FastAPI and to fetch the current
    configuration anywhere across the project.

    Returns:
        Settings: The application settings instance.
    """
    return Settings()
