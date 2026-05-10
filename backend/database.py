"""SQLite database configuration for RehabAI."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import os

DB_DIR = "/app/data" if os.path.exists("/app/data") else "."
URL_DATABASE = f"sqlite:///{DB_DIR}/RehabAI.db"

engine = create_engine(URL_DATABASE, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create a base class for all your models to inherit from
Base = declarative_base()
