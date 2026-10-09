import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker, declarative_base

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/stroke_twin"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def upgrade_demo_schema():
    """Add nullable fields used by newer local-demo versions without data loss."""
    tables = set(inspect(engine).get_table_names())
    additions = {
        "risk_assessments": {"ecg_result": "VARCHAR(40)"},
        "clinician_reviews": {
            "reviewer_name": "VARCHAR(80)",
            "disposition": "VARCHAR(24)",
        },
    }
    for table, columns in additions.items():
        if table not in tables:
            continue
        existing = {column["name"] for column in inspect(engine).get_columns(table)}
        for name, sql_type in columns.items():
            if name not in existing:
                with engine.begin() as connection:
                    connection.exec_driver_sql(
                        f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}"
                    )

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
