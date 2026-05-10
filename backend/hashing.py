"""Password hashing using bcrypt."""

from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def get_hashed_password(password):
    """Hash a password using bcrypt."""
    return pwd_context.hash(password)


def verify_password(plain_password, hashed_password):
    """Verify a password against a hashed password."""
    return pwd_context.verify(plain_password, hashed_password)
