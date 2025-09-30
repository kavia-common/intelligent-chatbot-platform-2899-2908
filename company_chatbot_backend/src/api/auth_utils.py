import os
import hmac
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field

try:
    import jwt  # PyJWT
except Exception:  # pragma: no cover
    jwt = None


class AuthSettings(BaseModel):
    """Settings for authentication loaded from environment."""
    secret_key: str = Field(default=os.environ.get("SECRET_KEY", ""), description="JWT secret key")
    algorithm: str = Field(default="HS256", description="JWT signing algorithm")
    access_token_expire_minutes: int = Field(default=int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "1440")))


def get_auth_settings() -> AuthSettings:
    return AuthSettings()


def _ensure_secret(secret: str):
    if not secret or len(secret) < 16:
        raise RuntimeError("SECRET_KEY is missing or too short. Set a strong key in the environment.")


# PUBLIC_INTERFACE
def hash_password(password: str) -> str:
    """Hash a password using HMAC-SHA256 with SECRET_KEY as salt. For demo use only."""
    secret = os.environ.get("SECRET_KEY", "")
    _ensure_secret(secret)
    return hmac.new(secret.encode("utf-8"), password.encode("utf-8"), hashlib.sha256).hexdigest()


# PUBLIC_INTERFACE
def verify_password(password: str, hashed: str) -> bool:
    """Verify password by recomputing demo HMAC hash. For production use bcrypt/argon2."""
    try:
        return hmac.compare_digest(hash_password(password), hashed)
    except Exception:
        return False


# PUBLIC_INTERFACE
def create_jwt_token(subject: str, expires_delta: Optional[timedelta] = None) -> Dict[str, Any]:
    """Create a signed JWT access token for the given subject (user id or email)."""
    st = get_auth_settings()
    _ensure_secret(st.secret_key)
    now = datetime.now(timezone.utc)
    exp = now + (expires_delta or timedelta(minutes=st.access_token_expire_minutes))
    payload = {
        "sub": subject,
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
        "type": "access",
    }
    if jwt is None:
        # Fallback unsigned token (development only)
        token = f"dev.{subject}.{int(exp.timestamp())}"
        return {"access_token": token, "expires_at": exp}
    token = jwt.encode(payload, st.secret_key, algorithm=st.algorithm)
    return {"access_token": token, "expires_at": exp}


# PUBLIC_INTERFACE
def decode_jwt_token(token: str) -> Dict[str, Any]:
    """Decode and verify a JWT token. Supports dev fallback."""
    st = get_auth_settings()
    if token.startswith("dev."):
        parts = token.split(".")
        if len(parts) == 3:
            return {"sub": parts[1], "exp": int(parts[2])}
        raise ValueError("Malformed dev token")
    if jwt is None:
        raise RuntimeError("JWT library not available. Install PyJWT or use dev token.")
    return jwt.decode(token, st.secret_key, algorithms=[st.algorithm])
