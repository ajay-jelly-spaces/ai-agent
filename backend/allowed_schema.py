"""
Allowed (approved) tables and columns for predefined queries and chat.
Only these tables and columns may be used in Mistral-generated SELECT queries.
"""

import re
from typing import Dict, List, Set

# Approved columns per table (single source of truth for users, bids, interpolated, otps, projects, random_test)
# interpolated may hold financial-style data; include both generic and financial columns
APPROVED_TABLE_COLUMNS: Dict[str, List[str]] = {
    "users": ["id", "username", "email", "created_at", "is_active"],
    "bids": ["id", "user_id", "project_id", "amount", "bid_time", "status"],
    "interpolated": [
        "id", "value", "timestamp", "source", "interpolated_at",
        "fiscaldateending", "grossprofit", "revenue", "netincome", "expenses", "created_at",
    ],
    "otps": ["id", "user_id", "otp_code", "expires_at", "used", "created_at"],
    "projects": ["id", "name", "description", "owner_id", "created_at", "status"],
    "random_test": ["id", "data", "timestamp", "test_type"],
}

APPROVED_TABLES: Set[str] = set(APPROVED_TABLE_COLUMNS.keys())


def get_allowed_schema_info(schema: str = "public") -> str:
    """
    Build schema description for Mistral containing ONLY approved tables and columns.
    Used so the model generates SELECTs that only reference allowed columns.
    """
    lines = [
        "ALLOWED SCHEMA (you may ONLY use these tables and columns in your SELECT queries):",
        "",
        "For schema metadata (table count, list tables) you may also use:",
        "- information_schema.tables (table_schema, table_name, table_type)",
        "- information_schema.columns (table_schema, table_name, column_name, data_type)",
        "",
        "Approved data tables and their columns:",
    ]
    for table, columns in APPROVED_TABLE_COLUMNS.items():
        cols = ", ".join(columns)
        lines.append(f'  Table: "{schema}"."{table}"  ->  columns: {cols}')
    lines.append("")
    lines.append(
        "RULES: If the user asks for specific columns, select ONLY those columns (if they are in the list above). "
        "If the user does not specify columns, you may use SELECT * or list all columns. "
        "Only query approved tables. Use double-quoted identifiers for schema and table names."
    )
    return "\n".join(lines)


def validate_sql_allowed(sql: str, schema: str = "public") -> None:
    """
    Validate that the SQL is a SELECT and only references approved tables and columns.
    Raises ValueError if not allowed.
    """
    sql_clean = sql.strip().rstrip(";").strip()
    sql_upper = sql_clean.upper()

    if not sql_upper.startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed.")

    # Forbidden keywords
    for kw in ("DELETE", "UPDATE", "INSERT", "DROP", "CREATE", "ALTER", "TRUNCATE", "EXECUTE", "CALL"):
        if kw in sql_upper:
            raise ValueError(f"Query must not contain {kw}.")

    # If query uses information_schema, allow (metadata questions)
    if "INFORMATION_SCHEMA" in sql_upper:
        return

    # Extract table name from FROM "schema"."table" or FROM schema.table or FROM table
    from_match = re.search(
        r'\bFROM\s+"(\w+)"\s*\.\s*"(\w+)"',
        sql_clean,
        re.IGNORECASE,
    )
    if from_match:
        table = from_match.group(2).lower()
    else:
        from_match = re.search(r'\bFROM\s+(\w+)\s*\.\s*(\w+)\b', sql_clean, re.IGNORECASE)
        if from_match:
            table = from_match.group(2).lower()
        else:
            from_match = re.search(r'\bFROM\s+"?(\w+)"?\s*(?:WHERE|ORDER|GROUP|LIMIT|$)', sql_clean, re.IGNORECASE)
            if from_match:
                table = from_match.group(1).strip('"').lower()
            else:
                raise ValueError("Could not determine which table the query uses.")

    if table not in APPROVED_TABLES:
        raise ValueError(f"Table '{table}' is not in the approved list. Approved: {sorted(APPROVED_TABLES)}.")

    # If SELECT *, allow (all columns for that table are approved)
    select_part = re.match(r"\s*SELECT\s+(.+?)\s+FROM", sql_clean, re.IGNORECASE | re.DOTALL)
    if not select_part:
        raise ValueError("Invalid SELECT form.")
    select_list = select_part.group(1).strip()
    if select_list.upper() == "*":
        return

    # Parse column list (simple: split by comma, strip quotes and aliases)
    allowed_cols = set(c.lower() for c in APPROVED_TABLE_COLUMNS[table])
    for part in select_list.split(","):
        part = part.strip()
        if not part:
            continue
        col = part.split()[0].strip('"').lower()
        if "." in col:
            col = col.split(".")[-1]
        if col not in allowed_cols:
            raise ValueError(
                f"Column '{col}' is not approved for table '{table}'. "
                f"Approved columns: {sorted(APPROVED_TABLE_COLUMNS[table])}."
            )
