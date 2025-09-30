from __future__ import annotations

from datetime import datetime
from sqlalchemy import Column, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship

from .database import Base


class UserModel(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=True)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    conversations = relationship("ConversationModel", back_populates="user")


class ConversationModel(Base):
    __tablename__ = "conversations"
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    title = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("UserModel", back_populates="conversations")
    messages = relationship("MessageModel", back_populates="conversation")


class MessageModel(Base):
    __tablename__ = "messages"
    id = Column(String, primary_key=True, index=True)
    role = Column(String, nullable=False)  # 'user' | 'assistant' | 'system'
    content = Column(Text, nullable=False)
    session_id = Column(String, ForeignKey("conversations.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    conversation = relationship("ConversationModel", back_populates="messages")
