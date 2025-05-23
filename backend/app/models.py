from sqlalchemy import Column, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id       = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email    = Column(String, unique=True, index=True, nullable=False)
    full_name= Column(String, nullable=True)
    hashed_password = Column(String, nullable=False)
    is_active     = Column(Boolean, default=True)
    totp_secret   = Column(String, nullable=True)  # để lưu key 2FA
