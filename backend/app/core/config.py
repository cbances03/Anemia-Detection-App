from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "Anemia Detection API"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = "development"

    HOST: str = "0.0.0.0"
    PORT: int = 8000

    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"

settings = Settings()
