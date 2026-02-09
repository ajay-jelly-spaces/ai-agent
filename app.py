"""
Streamlit app for displaying PostgreSQL data in a table format.
"""

from dotenv import load_dotenv

load_dotenv()

import os

import streamlit as st

from backend.db import run_select_query, get_tables
from backend.chat import ask_database


@st.cache_data(ttl=300)
def load_tables(schema: str = "public"):
    """Load table list from database with caching."""
    return get_tables(schema)


@st.cache_data(ttl=300)
def load_data(query: str):
    """
    Load data from PostgreSQL with caching.
    Cache expires after 5 minutes (ttl=300) to allow fresh data.
    """
    return run_select_query(query)


def render_data_tab(schema: str):
    """Render the data table loading tab."""
    table_options = ["-- Select a table --"]
    try:
        tables_df = load_tables(schema)
        if len(tables_df) > 0:
            table_options.extend(tables_df["table_name"].tolist())
    except Exception:
        pass

    selected_table = st.selectbox(
        "Select table",
        options=table_options,
        help="Choose a table to load its data",
    )

    if st.button("Load Data", type="primary", key="load_data_btn"):
        if selected_table == "-- Select a table --":
            st.error("Please select a table from the dropdown.")
        else:
            with st.spinner("Fetching data..."):
                try:
                    query = f'SELECT * FROM "{schema}"."{selected_table}"'
                    df = load_data(query)
                    row_count = len(df)

                    st.success(f"Loaded **{row_count:,}** row(s) from `{schema}.{selected_table}`.")

                    if row_count > 0:
                        st.dataframe(
                            df,
                            use_container_width=True,
                            height=400,
                        )
                    else:
                        st.info("The table is empty.")
                except ValueError as e:
                    st.error(f"❌ {e}")
                except Exception as e:
                    st.error(f"❌ An unexpected error occurred: {e}")


def render_chat_tab(schema: str):
    """Render the chat assistant tab."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            df = msg.get("dataframe")
            if df is not None and len(df) > 0:
                st.dataframe(df, use_container_width=True, height=200)

    if prompt := st.chat_input("Ask something about your database..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    df, summary = ask_database(prompt, schema)
                    st.markdown(summary)
                    if df is not None and len(df) > 0:
                        st.dataframe(df, use_container_width=True, height=200)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": summary,
                        "dataframe": df if df is not None and len(df) > 0 else None,
                    })
                except ValueError as e:
                    error_msg = f"❌ {e}"
                    st.markdown(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "dataframe": None,
                    })
                except Exception as e:
                    error_msg = f"❌ {e}"
                    st.markdown(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "dataframe": None,
                    })


def main():
    st.set_page_config(page_title="PostgreSQL Data Viewer", page_icon="📊", layout="wide")
    st.title("📊 PostgreSQL Data Viewer")

    db_name = os.environ.get("DB_NAME", "postgres")
    st.markdown(f"Database: **{db_name}** | Select a table or ask questions in natural language.")

    schema = st.sidebar.text_input("Schema", value="public", help="Schema to list tables from")

    tab1, tab2 = st.tabs(["📋 Data", "💬 Chat Assistant"])

    with tab1:
        render_data_tab(schema)

    with tab2:
        render_chat_tab(schema)


if __name__ == "__main__":
    main()
