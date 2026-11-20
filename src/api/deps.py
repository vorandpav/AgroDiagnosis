"""FastAPI dependencies for lifespan-created application resources."""

from typing import Annotated

import asyncpg
from fastapi import Depends, Request
from faststream.rabbit import RabbitBroker

from src.config import Settings


def get_settings(request: Request) -> Settings:
    """Return application settings from app state."""
    return request.app.state.settings


def get_db_pool(request: Request) -> asyncpg.Pool:
    """Return the shared asyncpg pool from app state."""
    return request.app.state.db_pool


def get_broker(request: Request) -> RabbitBroker:
    """Return the connected RabbitMQ broker from app state."""
    return request.app.state.broker


SettingsDep = Annotated[Settings, Depends(get_settings)]
PoolDep = Annotated[asyncpg.Pool, Depends(get_db_pool)]
BrokerDep = Annotated[RabbitBroker, Depends(get_broker)]
