from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AI-PARK"
    app_version: str = "1.0.0"
    database_url: str = "sqlite:///./data/aipark.db"
    frontend_url: str = "https://ai-park.vercel.app"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
