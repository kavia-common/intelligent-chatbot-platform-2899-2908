from __future__ import annotations

from datetime import datetime
from sqlalchemy import Column, String, DateTime, Text, ForeignKey, LargeBinary


from .database import Base


class UserModel(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=True)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class ChatMessageModel(Base):
    __tablename__ = "chat_messages"
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    message = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class DocumentModel(Base):
    __tablename__ = "documents"
    id = Column(String, primary_key=True, index=True)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    # Embedding may be stored via pgvector extension as a 'vector' type. For portable fallback, we keep bytes.
    # In production, map to pgvector via SQLAlchemy plugin or use BYTEA as done here.
    embedding = Column(LargeBinary, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
