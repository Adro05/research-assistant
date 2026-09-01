"""
Declarative base for SQLAlchemy ORM models.

No tables/entities are defined yet. Models (Project, Document, etc., per
ARCHITECTURE.md §6-7) will be added in a later, explicitly-requested
milestone and should inherit from `Base`.
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all ORM models."""