"""
Chat assistant for natural language to SQL queries.
Uses RAG system where AI has no direct database access.
"""

import os
from typing import Optional

import pandas as pd

from backend.db import run_select_query
from backend.allowed_schema import get_allowed_schema_info, validate_sql_allowed
from backend.rag_knowledge import RAGChatInterface


SYSTEM_PROMPT = """You are a helpful PostgreSQL assistant. Given the ALLOWED schema and a user question, you must respond with ONLY a valid PostgreSQL SELECT query. No explanations, no markdown, no backticks - just the raw SQL.

You may ONLY use tables and columns listed in the ALLOWED SCHEMA. If the user asks for specific columns, select only those columns (and only if they appear in the approved column list for that table). If the user does not specify columns, you may use SELECT * or list all approved columns for that table.

You may also use information_schema.tables and information_schema.columns for metadata (e.g. table count, list tables).

If the question cannot be answered with the allowed schema, respond with: ERROR: <brief reason>."""

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
    Process a natural language question using RAG system.
    AI has no direct database access - only works with knowledge and secure APIs.
    
    Returns:
        Tuple of (DataFrame or None, response text)
    """
    try:
        # Use RAG interface with fresh knowledge base
        rag_interface = RAGChatInterface(schema)
        # Force rebuild to get latest table information
        rag_interface.assistant._initialize_knowledge_base(force_rebuild=True)
        return rag_interface.ask_database(question, schema)
    except Exception as e:
        # Fallback to original method if RAG fails
        try:
            # Use allowed schema so Mistral only generates SELECTs with approved tables/columns
            schema_info = get_allowed_schema_info(schema)
            sql = get_sql_from_question(question, schema_info)
            if not sql.strip().upper().startswith("SELECT"):
                raise ValueError("Generated query must be a SELECT statement.")
            validate_sql_allowed(sql, schema)
            df = run_select_query(sql)
            summary = summarize_results(df)
            return df, summary
        except Exception as fallback_error:
            return None, str(fallback_error)
