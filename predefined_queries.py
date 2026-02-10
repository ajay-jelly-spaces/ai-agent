# predefined_queries.py
# Usage:
#   python predefined_queries.py "How many tables are there in the database?"
#   python predefined_queries.py "List all tables" --schema public
#   python predefined_queries.py "Columns for table users" --schema public
#
# Requires: backend.db and backend.allowed_schema, and env vars for DB.

import argparse
import re
from typing import Dict, List, Optional, Tuple

from backend.db import run_select_query, get_table_columns
from backend.allowed_schema import APPROVED_TABLE_COLUMNS as _ALLOWED_COLUMNS

# Extended approved list for this script (includes financial_* if present in DB)
APPROVED_TABLE_COLUMNS: Dict[str, List[str]] = dict(_ALLOWED_COLUMNS)
APPROVED_TABLE_COLUMNS.update({
    "financial_data": ["id", "fiscaldateending", "grossprofit", "revenue", "netincome", "created_at"],
    "financial_reports": ["id", "fiscaldateending", "grossprofit", "revenue", "expenses", "quarter", "year"],
})

# ---------------------------
# Predefined SQL templates
# ---------------------------

PREDEFINED_QUERIES: Dict[str, Dict] = {
    "table_count": {
        "patterns": [
            r"\bhow many tables\b",
            r"\bnumber of tables\b",
            r"\bcount tables\b",
        ],
        "sql": """
            SELECT COUNT(*) AS table_count
            FROM information_schema.tables
            WHERE table_schema = :schema
              AND table_type = 'BASE TABLE';
        """,
        "param_builder": lambda schema, table, limit: {"schema": schema},
    },
    "list_tables": {
        "patterns": [
            r"\blist (all )?tables\b",
            r"\bshow (all )?tables\b",
            r"\bwhat tables (exist|are there)\b",
        ],
        "sql": """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = :schema
              AND table_type = 'BASE TABLE'
            ORDER BY table_name;
        """,
        "param_builder": lambda schema, table, limit: {"schema": schema},
    },
    "column_count": {
        "patterns": [
            r"\bhow many columns\b",
            r"\bnumber of columns\b",
            r"\bcount columns\b",
        ],
        "sql": """
            SELECT COUNT(*) AS column_count
            FROM information_schema.columns
            WHERE table_schema = :schema
              AND table_name = :table;
        """,
        "param_builder": lambda schema, table, limit: {"schema": schema, "table": table},
        "requires_table": True,
    },
    "list_columns": {
        "patterns": [
            r"\blist columns\b",
            r"\bshow columns\b",
            r"\bcolumns for table\b",
            r"\bdescribe table\b",
        ],
        "sql": """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = :schema
              AND table_name = :table
            ORDER BY ordinal_position;
        """,
        "param_builder": lambda schema, table, limit: {"schema": schema, "table": table},
        "requires_table": True,
    },
    "top_n_rows": {
        "patterns": [
            r"\btop \d+\b",
            r"\blimit \d+\b",
            r"\bshow \d+ rows\b",
        ],
        # NOTE: identifiers cannot be bound safely via :params; we validate and interpolate
        "sql_builder": lambda schema, table, limit, question: f'SELECT * FROM "{schema}"."{table}" LIMIT {limit};',
        "requires_table": True,
        "requires_limit": True,
    },
    
    # Table-specific queries with column security
    "users_select": {
        "patterns": [
            r"\busers\b.*\b(select|show|get)\b",
            r"\b(select|show|get)\b.*\busers\b",
            r"\busers\b.*\bdata\b",
        ],
        "sql_builder": lambda schema, table, limit, question: build_secure_select_query(schema, "users", extract_columns_from_question(question), limit),
        "requires_table": True,
        "table_specific": "users",
    },
    "bids_select": {
        "patterns": [
            r"\bbids\b.*\b(select|show|get)\b",
            r"\b(select|show|get)\b.*\bbids\b",
            r"\bbids\b.*\bdata\b",
        ],
        "sql_builder": lambda schema, table, limit, question: build_secure_select_query(schema, "bids", extract_columns_from_question(question), limit),
        "requires_table": True,
        "table_specific": "bids",
    },
    "interpolated_select": {
        "patterns": [
            r"\binterpolated\b.*\b(select|show|get)\b",
            r"\b(select|show|get)\b.*\binterpolated\b",
            r"\binterpolated\b.*\bdata\b",
        ],
        "sql_builder": lambda schema, table, limit, question: build_secure_select_query(schema, "interpolated", extract_columns_from_question(question), limit),
        "requires_table": True,
        "table_specific": "interpolated",
    },
    "otps_select": {
        "patterns": [
            r"\botps\b.*\b(select|show|get)\b",
            r"\b(select|show|get)\b.*\botps\b",
            r"\botps\b.*\bdata\b",
        ],
        "sql_builder": lambda schema, table, limit, question: build_secure_select_query(schema, "otps", extract_columns_from_question(question), limit),
        "requires_table": True,
        "table_specific": "otps",
    },
    "projects_select": {
        "patterns": [
            r"\bprojects\b.*\b(select|show|get)\b",
            r"\b(select|show|get)\b.*\bprojects\b",
            r"\bprojects\b.*\bdata\b",
        ],
        "sql_builder": lambda schema, table, limit, question: build_secure_select_query(schema, "projects", extract_columns_from_question(question), limit),
        "requires_table": True,
        "table_specific": "projects",
    },
    "random_test_select": {
        "patterns": [
            r"\brandom_test\b.*\b(select|show|get)\b",
            r"\b(select|show|get)\b.*\brandom_test\b",
            r"\brandom_test\b.*\bdata\b",
        ],
        "sql_builder": lambda schema, table, limit, question: build_secure_select_query(schema, "random_test", extract_columns_from_question(question), limit),
        "requires_table": True,
        "table_specific": "random_test",
    },
    "financial_data_select": {
        "patterns": [
            r"\bfinancial_data\b.*\b(select|show|get)\b",
            r"\b(select|show|get)\b.*\bfinancial_data\b",
            r"\bfinancial_data\b.*\bdata\b",
            r"\bgrossprofit\b",
            r"\bfiscaldateending\b",
        ],
        "sql_builder": lambda schema, table, limit, question: build_secure_select_query(schema, "financial_data", extract_columns_from_question(question), limit),
        "requires_table": True,
        "table_specific": "financial_data",
    },
    "financial_reports_select": {
        "patterns": [
            r"\bfinancial_reports\b.*\b(select|show|get)\b",
            r"\b(select|show|get)\b.*\bfinancial_reports\b",
            r"\bfinancial_reports\b.*\bdata\b",
            r"\bgrossprofit\b.*\brevenue\b",
            r"\bfiscal\s+reports\b",
        ],
        "sql_builder": lambda schema, table, limit, question: build_secure_select_query(schema, "financial_reports", extract_columns_from_question(question), limit),
        "requires_table": True,
        "table_specific": "financial_reports",
    },
}


# ---------------------------
# Helpers
# ---------------------------

IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def extract_columns_from_question(user_text: str) -> List[str]:
    """
    Extract column names from user question based on approved columns.
    Only returns columns that are in the approved whitelist.
    """
    user_text_lower = user_text.lower()
    extracted_columns = []
    
    # Check for each approved table and its columns
    for table_name, approved_columns in APPROVED_TABLE_COLUMNS.items():
        if table_name.lower() in user_text_lower:
            for column in approved_columns:
                # Check if column name is mentioned in the question
                if column.lower() in user_text_lower:
                    extracted_columns.append(column)
    
    # If no specific columns mentioned, return all approved columns for the detected table
    if not extracted_columns:
        for table_name, approved_columns in APPROVED_TABLE_COLUMNS.items():
            if table_name.lower() in user_text_lower:
                return approved_columns
    
    return extracted_columns


def build_secure_select_query(schema: str, table: str, columns: List[str], limit: Optional[int] = None) -> str:
    """
    Build a secure SELECT query using only approved columns that exist in the DB.
    Uses actual table columns from information_schema so we never select non-existent columns.
    """
    if table not in APPROVED_TABLE_COLUMNS:
        raise ValueError(f"Table '{table}' is not in the approved list")

    approved_columns = APPROVED_TABLE_COLUMNS[table]
    try:
        actual_columns = get_table_columns(schema, table)
    except Exception:
        actual_columns = []

    # Only use columns that exist in the DB and are approved
    allowed_actual = [c for c in actual_columns if c in approved_columns]
    if not allowed_actual and actual_columns:
        # DB has columns but none match our approved list - use approved list as-is and let execution fail with clear error
        allowed_actual = [c for c in approved_columns if c in actual_columns]
    if not allowed_actual:
        allowed_actual = approved_columns  # fallback if we couldn't fetch (e.g. offline)

    if not columns:
        columns = allowed_actual
    else:
        # Requested columns: must be approved and exist in DB
        columns = [c for c in columns if c in approved_columns and c in actual_columns]
        if not columns:
            columns = allowed_actual

    if not columns:
        raise ValueError(
            f"No columns available for table '{table}'. "
            f"Approved: {approved_columns}. DB has: {actual_columns}."
        )

    column_list = ", ".join([f'"{col}"' for col in columns])
    query = f'SELECT {column_list} FROM "{schema}"."{table}"'
    if limit:
        query += f' LIMIT {limit}'
    return query + ";"


def validate_select_only_query(sql: str) -> bool:
    """
    Validate that the query is SELECT-only for maximum security.
    """
    sql_upper = sql.strip().upper()
    
    # Must start with SELECT
    if not sql_upper.startswith('SELECT'):
        return False
    
    # Forbidden keywords for security
    forbidden_keywords = [
        'DROP', 'DELETE', 'UPDATE', 'INSERT', 'CREATE', 'ALTER', 
        'TRUNCATE', 'GRANT', 'REVOKE', 'COMMIT', 'ROLLBACK', 
        'EXECUTE', 'CALL', 'MERGE', 'UNION', 'INTERSECT', 'EXCEPT'
    ]
    
    for keyword in forbidden_keywords:
        if keyword in sql_upper:
            return False
    
    # Only allow safe SQL patterns
    allowed_patterns = [
        r'^SELECT\s+.*\s+FROM\s+',
        r'^SELECT\s+.*\s+FROM\s+.*\s+WHERE\s+',
        r'^SELECT\s+.*\s+FROM\s+.*\s+ORDER\s+BY\s+',
        r'^SELECT\s+.*\s+FROM\s+.*\s+LIMIT\s+',
        r'^SELECT\s+.*\s+FROM\s+.*\s+GROUP\s+BY\s+',
        r'^SELECT\s+.*\s+FROM\s+.*\s+HAVING\s+'
    ]
    
    return any(re.match(pattern, sql_upper, re.IGNORECASE) for pattern in allowed_patterns)


def extract_table_name(user_text: str) -> Optional[str]:
    """
    Attempts to extract a table name from phrases like:
    - "columns for table users"
    - "describe table orders"
    - "columns for users"
    Also checks for our approved table names.
    """
    # First try the existing pattern
    m = re.search(r"\btable\s+([A-Za-z_][A-Za-z0-9_]*)\b", user_text, re.IGNORECASE)
    if m:
        return m.group(1)
    
    # Check for approved table names directly
    user_text_lower = user_text.lower()
    for table_name in APPROVED_TABLE_COLUMNS.keys():
        if table_name.lower() in user_text_lower:
            return table_name
    
    m = re.search(r"\bfor\s+([A-Za-z_][A-Za-z0-9_]*)\b", user_text, re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def extract_limit(user_text: str) -> Optional[int]:
    m = re.search(r"\b(top|limit|show)\s+(\d+)\b", user_text, re.IGNORECASE)
    if m:
        return int(m.group(2))
    return None


def validate_identifier(name: str, kind: str) -> str:
    if not name or not IDENT_RE.match(name):
        raise ValueError(f"Invalid {kind} identifier: {name!r}")
    return name


def match_predefined(user_text: str) -> Tuple[Optional[str], Optional[dict]]:
    text = user_text.strip().lower()

    for key, spec in PREDEFINED_QUERIES.items():
        for pat in spec["patterns"]:
            if re.search(pat, text, re.IGNORECASE):
                return key, spec
    return None, None


# ---------------------------
# Main execution
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Run predefined DB metadata queries (no LLM).")
    parser.add_argument("question", type=str, help="User question, e.g. 'How many tables are there?'")
    parser.add_argument("--schema", type=str, default="public", help="Schema name (default: public)")
    parser.add_argument("--table", type=str, default=None, help="Table name (optional; can be inferred)")
    parser.add_argument("--limit", type=int, default=None, help="Row limit for top-N (optional; can be inferred)")
    args = parser.parse_args()

    q = args.question
    schema = validate_identifier(args.schema, "schema")

    # Find which predefined query this is
    key, spec = match_predefined(q)
    if not key:
        raise SystemExit(
            "No predefined query matched.\n"
            "Try: 'How many tables are there?', 'List all tables', 'List columns for table X', 'Top 10 rows for table X'."
        )

    # Infer table/limit if needed
    table = args.table or extract_table_name(q)
    limit = args.limit or extract_limit(q)

    if spec.get("requires_table"):
        if not table:
            raise SystemExit("This query requires a table name. Provide --table <name> or include it in the question.")
        table = validate_identifier(table, "table")

    if spec.get("requires_limit"):
        if not limit:
            raise SystemExit("This query requires a numeric limit (e.g., 'top 10'). Provide --limit <n> or include it.")
        if limit <= 0 or limit > 10000:
            raise SystemExit("Limit out of allowed range (1..10000).")
        # validate interpolated identifiers
        _ = validate_identifier(schema, "schema")
        _ = validate_identifier(table, "table")

    # Build and run SQL
    if "sql_builder" in spec:
        sql = spec["sql_builder"](schema, table, limit, q)
        df = run_select_query(sql)
    else:
        sql = spec["sql"]
        params = spec["param_builder"](schema, table, limit)
        df = run_select_query(sql, params=params)

    # Print result
    print(f"\nMatched predefined query: {key}")
    print(df.to_string(index=False))
    print()

if __name__ == "__main__":
    main()
