"""
Chat assistant for natural language to SQL queries.
Uses Mistral API to convert user questions to SQL and format responses.
"""

import os
from typing import Optional

import pandas as pd

from backend.db import get_schema_info, run_select_query


SYSTEM_PROMPT = """You are a helpful PostgreSQL assistant. Given a database schema and a user question, you must respond with ONLY a valid PostgreSQL SELECT query. No explanations, no markdown, no backticks - just the raw SQL.

You can query:
1. User tables (listed in the schema) for data questions.
2. information_schema.tables for table count, list of tables - use table_schema, table_name, table_type.
3. information_schema.columns for column metadata.

If the question cannot be answered at all, respond with: ERROR: <brief reason>."""

SUMMARY_PROMPT = """Summarize these query results in 1-2 sentences for the user. Be concise and conversational."""


def _extract_text(content) -> str:
    """Extract text from Mistral response content (string or list of chunks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            c.get("text", "") for c in content if isinstance(c, dict) and "text" in c
        )
    return str(content)


def _get_mistral_client():
    """Get Mistral client."""
    try:
        from mistralai import Mistral
    except ImportError:
        raise ValueError(
            "mistralai package not installed. Run: pip install mistralai"
        )

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError(
            "MISTRAL_API_KEY not set. Add it to your .env file to use chat assistance."
        )

    return Mistral(api_key=api_key)


def get_sql_from_question(question: str, schema_info: str) -> str:
    """Convert natural language question to SQL using Mistral."""
    client = _get_mistral_client()
    model = os.getenv("MISTRAL_MODEL", "mistral-small-latest")

    response = client.chat.complete(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{schema_info}\n\nUser question: {question}"},
        ],
        temperature=0,
    )
    content = response.choices[0].message.content
    sql = _extract_text(content).strip()

    if sql.upper().startswith("ERROR:"):
        raise ValueError(sql)

    # Remove markdown code blocks if present
    if sql.startswith("```"):
        lines = sql.split("\n")
        sql = "\n".join(lines[1:-1]) if len(lines) > 2 else sql

    return sql


def summarize_results(df: pd.DataFrame) -> str:
    """Generate a brief natural language summary of query results using Mistral."""
    try:
        client = _get_mistral_client()
    except ValueError:
        return f"Query returned {len(df)} row(s)."

    if df.empty:
        return "The query returned no rows."

    sample = df.head(5).to_string()
    prompt = f"Query results (first 5 rows):\n{sample}\n\nTotal rows: {len(df)}\n\n{SUMMARY_PROMPT}"

    try:
        model = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
        response = client.chat.complete(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        content = response.choices[0].message.content
        return _extract_text(content).strip()
    except Exception:
        return f"Query returned {len(df)} row(s)."


def ask_database(
    question: str,
    schema: str = "public",
) -> tuple[Optional[pd.DataFrame], str]:
    """
    Process a natural language question: generate SQL, execute, return results and response.

    Returns:
        Tuple of (DataFrame or None, response text)
    """
    schema_info = get_schema_info(schema)

    sql = get_sql_from_question(question, schema_info)

    if not sql.strip().upper().startswith("SELECT"):
        raise ValueError("Generated query must be a SELECT statement.")

    df = run_select_query(sql)
    summary = summarize_results(df)

    return df, summary
