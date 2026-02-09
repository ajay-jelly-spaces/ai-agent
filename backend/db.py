"""
Database module for PostgreSQL connections and query execution.
Reads credentials from environment variables.
"""

import os
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError


def get_connection_string() -> str:
    """Build PostgreSQL connection string from environment variables."""
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME", "jellyspace_database")
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD", "postgres")

    return f"postgresql://{user}:{password}@{host}:{port}/{name}"


def run_select_query(
    query: str,
    params: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Execute a SELECT query and return results as a Pandas DataFrame.

    Args:
        query: SQL SELECT query string.
        params: Optional dict of parameters for parameterized queries.

    Returns:
        DataFrame containing the query results.

    Raises:
        ValueError: If required env vars are missing or query execution fails.
    """
    # Validate required credentials
    password = os.getenv("DB_PASSWORD")
    if password is None or password == "":
        raise ValueError(
            "Database password is not set. Please set DB_PASSWORD in your environment."
        )

    try:
        engine = create_engine(
            get_connection_string(),
            connect_args={"connect_timeout": 10},
        )
    except Exception as e:
        raise ValueError(f"Failed to create database connection: {e}") from e

    try:
        with engine.connect() as conn:
            if params:
                df = pd.read_sql(text(query), conn, params=params)
            else:
                df = pd.read_sql(text(query), conn)
        return df
    except SQLAlchemyError as e:
        error_msg = str(e.orig) if hasattr(e, "orig") else str(e)
        raise ValueError(f"Database query failed: {error_msg}") from e
    except Exception as e:
        raise ValueError(f"Unexpected error while executing query: {e}") from e


def get_tables(schema: str = "public") -> pd.DataFrame:
    """
    Get list of tables from the database for the given schema.

    Args:
        schema: Schema name (default: public).

    Returns:
        DataFrame with columns: table_schema, table_name, table_type.
    """
    query = """
        SELECT table_schema, table_name, table_type
        FROM information_schema.tables
        WHERE table_schema = :schema
          AND table_type = 'BASE TABLE'
        ORDER BY table_schema, table_name
    """
    return run_select_query(query, params={"schema": schema})


def get_schema_info(schema: str = "public") -> str:
    """
    Get schema description (tables and columns) for LLM context.
    Includes information_schema so the LLM can answer meta-questions.

    Returns:
        Human-readable string describing the database schema.
    """
    info_schema_note = f"""
AVAILABLE FOR SCHEMA METADATA (table count, list tables, etc.):
- information_schema.tables: table_schema, table_name, table_type
  Use for: "how many tables", "list all tables", "what tables exist"
  Example: SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = '{schema}' AND table_type = 'BASE TABLE'
- information_schema.columns: table_schema, table_name, column_name, data_type
  Use for: column counts, column listings
"""

    query = """
        SELECT t.table_name, c.column_name, c.data_type
        FROM information_schema.tables t
        JOIN information_schema.columns c
          ON t.table_schema = c.table_schema AND t.table_name = c.table_name
        WHERE t.table_schema = :schema
          AND t.table_type = 'BASE TABLE'
        ORDER BY t.table_name, c.ordinal_position
    """
    df = run_select_query(query, params={"schema": schema})

    lines = ["Schema for PostgreSQL database (schema: " + schema + "):", info_schema_note]
    if df.empty:
        lines.append(f"\nNo user tables in schema '{schema}'.")
    else:
        current_table = None
        for _, row in df.iterrows():
            if row["table_name"] != current_table:
                current_table = row["table_name"]
                lines.append(f"\nTable: {schema}.{current_table}")
            lines.append(f"  - {row['column_name']} ({row['data_type']})")

    return "".join(lines).strip()
