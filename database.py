from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = "postgresql://postgres:postgres@127.0.0.1:5433/ai_chat_db"

engine = create_engine(
    DATABASE_URL,
    echo=False,         # set True only while debugging
    pool_size=10,       # max persistent connections
    max_overflow=20,    # extra connections under heavy load
    pool_timeout=30,    # wait time before raising error
    pool_recycle=1800,  # recycle connections every 30 min
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()