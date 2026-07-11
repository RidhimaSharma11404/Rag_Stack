from sqlalchemy import Column, Integer, String, DateTime, Text
from sqlalchemy.sql import func
from database import Base


class User(Base):
    __tablename__ = "users"

    id         = Column(Integer, primary_key=True, index=True)
    username   = Column(String(50), unique=True, nullable=False, index=True)
    password   = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Message(Base):
    __tablename__ = "messages"

    id            = Column(Integer, primary_key=True, index=True)
    username      = Column(String(50), nullable=False, index=True)
    user_message  = Column(Text, nullable=False)
    bot_response  = Column(Text, nullable=False)
    score         = Column(String(10))
    db_used       = Column(String(20))
    response_time = Column(String(10))      # ← seconds taken to search
    created_at    = Column(DateTime(timezone=True), server_default=func.now())