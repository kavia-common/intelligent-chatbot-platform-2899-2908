from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

from sqlalchemy import select, func
from sqlalchemy.orm import Session

from .models import UserModel, ChatMessageModel, DocumentModel


# PUBLIC_INTERFACE
def create_user(db: Session, email: str, password_hash: str, name: Optional[str] = None) -> UserModel:
    """Create a new user row."""
    user = UserModel(
        id=str(uuid.uuid4()),
        email=email.lower(),
        name=name,
        password_hash=password_hash,
        created_at=datetime.utcnow(),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


# PUBLIC_INTERFACE
def get_user_by_email(db: Session, email: str) -> Optional[UserModel]:
    """Fetch a user by email."""
    stmt = select(UserModel).where(func.lower(UserModel.email) == email.lower())
    return db.execute(stmt).scalars().first()


# PUBLIC_INTERFACE
def add_chat_message(db: Session, user_id: str, message_text: str) -> ChatMessageModel:
    """Insert a chat message for a user."""
    msg = ChatMessageModel(
        id=str(uuid.uuid4()),
        user_id=user_id,
        message=message_text,
        created_at=datetime.utcnow(),
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg


# PUBLIC_INTERFACE
def list_user_messages(db: Session, user_id: str, limit: int = 100) -> List[ChatMessageModel]:
    """List recent chat messages for a user."""
    stmt = (
        select(ChatMessageModel)
        .where(ChatMessageModel.user_id == user_id)
        .order_by(ChatMessageModel.created_at.desc())
        .limit(limit)
    )
    return list(db.execute(stmt).scalars().all())


# PUBLIC_INTERFACE
def ingest_documents(db: Session, docs: List[Dict[str, Any]]) -> int:
    """Bulk insert documents with optional embedding vectors."""
    count = 0
    for d in docs:
        title = d.get("title") or (d.get("source") or "manual")
        content = d.get("content") or d.get("text") or ""
        if not content.strip():
            continue
        embedding = d.get("embedding")  # list[float] or bytes
        row = DocumentModel(
            id=str(uuid.uuid4()),
            title=title,
            content=content,
            embedding=embedding,
            created_at=datetime.utcnow(),
        )
        db.add(row)
        count += 1
    db.commit()
    return count


# PUBLIC_INTERFACE
def list_documents(db: Session, limit: int = 100) -> List[DocumentModel]:
    """List recent documents."""
    stmt = select(DocumentModel).order_by(DocumentModel.created_at.desc()).limit(limit)
    return list(db.execute(stmt).scalars().all())
