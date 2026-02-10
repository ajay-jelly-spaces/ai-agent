"""
RAG Knowledge Base for Database Queries
AI has no direct database access - only works with vectorized schema knowledge.
"""

import os
import re
from typing import Dict, List, Optional, Tuple
import json
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from backend.db import get_schema_info


class _UseMistralFallback(Exception):
    """Raise to signal chat should use Mistral fallback (e.g. for filtered queries)."""


@dataclass
class SchemaKnowledge:
    """Knowledge item about database schema."""
    content: str
    metadata: Dict
    embedding: Optional[np.ndarray] = None


class DatabaseKnowledgeBase:
    """Vector knowledge base for database schema information."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.knowledge_items: List[SchemaKnowledge] = []
        self.is_indexed = False
    
    def extract_schema_knowledge(self, schema: str = "public") -> List[SchemaKnowledge]:
        """Extract structured knowledge from database schema."""
        knowledge_items = []
        
        # Get basic schema info
        schema_info = get_schema_info(schema)
        
        # Add schema overview
        knowledge_items.append(SchemaKnowledge(
            content=f"Database schema overview for {schema}: {schema_info}",
            metadata={"type": "schema_overview", "schema": schema}
        ))
        
        # Add knowledge about approved tables and their columns
        try:
            from predefined_queries import APPROVED_TABLE_COLUMNS
            
            for table_name, columns in APPROVED_TABLE_COLUMNS.items():
                # Add table knowledge
                columns_desc = ", ".join(columns)
                knowledge_items.append(SchemaKnowledge(
                    content=f"Table {schema}.{table_name} has approved columns: {columns_desc}",
                    metadata={
                        "type": "approved_table_definition",
                        "schema": schema,
                        "table": table_name,
                        "approved_columns": columns
                    }
                ))
                
                # Add individual column knowledge
                for column in columns:
                    knowledge_items.append(SchemaKnowledge(
                        content=f"Column {column} is available in table {schema}.{table_name}",
                        metadata={
                            "type": "approved_column_definition",
                            "schema": schema,
                            "table": table_name,
                            "column": column
                        }
                    ))
        
        except Exception as e:
            print(f"Error adding approved table knowledge: {e}")
        
        # Add predefined query patterns knowledge
        predefined_knowledge = [
            "Use predefined patterns for secure queries on approved tables only",
            "Approved tables: users, bids, interpolated, otps, projects, random_test, financial_data, financial_reports",
            "Column-level security: Only approved columns can be selected",
            "Financial data available in financial_data and financial_reports tables",
            "financial_data table columns: id, fiscaldateending, grossprofit, revenue, netincome, created_at",
            "financial_reports table columns: id, fiscaldateending, grossprofit, revenue, expenses, quarter, year",
            "Table count queries use information_schema.tables",
            "Column information from information_schema.columns",
            "Table listings use information_schema.tables with table_type = 'BASE TABLE'",
            "Column listings use information_schema.columns with ordinal_position",
            "Security: No raw SQL generation, only predefined patterns allowed"
        ]
        
        for i, content in enumerate(predefined_knowledge):
            knowledge_items.append(SchemaKnowledge(
                content=content,
                metadata={"type": "security_patterns", "schema": schema, "index": i}
            ))
        
        return knowledge_items
    
    def build_index(self, schema: str = "public"):
        """Build vector index from schema knowledge."""
        print("Extracting schema knowledge...")
        self.knowledge_items = self.extract_schema_knowledge(schema)
        
        print("Generating embeddings...")
        contents = [item.content for item in self.knowledge_items]
        embeddings = self.model.encode(contents, show_progress_bar=True)
        
        for i, item in enumerate(self.knowledge_items):
            item.embedding = embeddings[i]
        
        self.is_indexed = True
        print(f"Indexed {len(self.knowledge_items)} knowledge items")
    
    def retrieve_relevant_knowledge(self, query: str, top_k: int = 5) -> List[SchemaKnowledge]:
        """Retrieve most relevant schema knowledge for a query."""
        if not self.is_indexed:
            raise ValueError("Knowledge base not indexed. Call build_index() first.")
        
        query_embedding = self.model.encode([query])
        
        similarities = []
        for item in self.knowledge_items:
            sim = cosine_similarity(query_embedding, [item.embedding])[0][0]
            similarities.append(sim)
        
        # Get top-k most similar items
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [self.knowledge_items[i] for i in top_indices]


class SecureQueryInterface:
    """Secure interface that handles database queries without exposing direct access to AI."""
    
    def __init__(self):
        self.predefined_queries = self._load_predefined_queries()
    
    def _load_predefined_queries(self):
        """Load predefined queries from existing system."""
        try:
            # Import from the root directory
            import sys
            import os
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if root_dir not in sys.path:
                sys.path.insert(0, root_dir)
            
            from predefined_queries import PREDEFINED_QUERIES, match_predefined, extract_table_name, extract_limit
            return {
                'PREDEFINED_QUERIES': PREDEFINED_QUERIES,
                'match_predefined': match_predefined,
                'extract_table_name': extract_table_name,
                'extract_limit': extract_limit
            }
        except ImportError as e:
            print(f"Could not import predefined_queries from {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}: {e}")
            return None
    
    def execute_query_safely(self, question: str, schema: str = "public"):
        """Execute query using predefined patterns only - SELECT-only, no direct DB access for AI."""
        if not self.predefined_queries:
            raise ValueError("Predefined queries not available")
        
        # Try to match predefined patterns
        key, spec = self.predefined_queries['match_predefined'](question)
        if not key:
            raise ValueError("No predefined pattern matched. AI cannot generate custom queries for security.")

        # If question asks for a filter (WHERE clause), skip predefined so Mistral fallback can generate SELECT...WHERE
        q_lower = question.lower()
        has_filter = (
            " where " in q_lower or " where\t" in q_lower
            or "fiscaldateending" in q_lower
            or " is 20" in q_lower  # e.g. "is 2023-09-30"
        )
        if has_filter and spec.get("sql_builder") and spec.get("table_specific"):
            raise _UseMistralFallback(
                "Question asks for a filtered query; Mistral will generate SELECT with WHERE."
            )

        # Execute the predefined query
        try:
            from predefined_queries import validate_identifier, validate_select_only_query
            from backend.db import run_select_query
        except ImportError as e:
            raise ValueError(f"Could not import required modules: {e}")
        
        # Extract parameters
        table = None
        limit = None
        
        if spec.get("requires_table"):
            table = self.predefined_queries['extract_table_name'](question)
            if not table:
                raise ValueError("This query requires a table name")
            table = validate_identifier(table, "table")
        
        if spec.get("requires_limit"):
            limit = self.predefined_queries['extract_limit'](question)
            if not limit:
                raise ValueError("This query requires a numeric limit")
            if limit <= 0 or limit > 10000:
                raise ValueError("Limit out of allowed range (1..10000)")
        
        # Execute query with SELECT-only validation
        try:
            if "sql_builder" in spec:
                sql = spec["sql_builder"](schema, table, limit, question)
            else:
                sql = spec["sql"]
                params = spec["param_builder"](schema, table, limit)
            
            # CRITICAL: Validate that the generated SQL is SELECT-only
            if not validate_select_only_query(sql):
                raise ValueError(f"Generated query is not SELECT-only: {sql}")
            
            # Execute the validated SELECT query
            if "params" in locals():
                df = run_select_query(sql, params=params)
            else:
                df = run_select_query(sql)
            
            return df, key, sql
        except Exception as e:
            raise ValueError(f"Query execution failed: {e}")


class RAGDatabaseAssistant:
    """Main RAG system - AI only interacts with knowledge, never with database directly."""
    
    def __init__(self, schema: str = "public", model_name: str = "all-MiniLM-L6-v2"):
        self.schema = schema
        self.knowledge_base = DatabaseKnowledgeBase(model_name)
        self.secure_interface = SecureQueryInterface()
        
        # Initialize knowledge base - force rebuild to include new tables
        self._initialize_knowledge_base(force_rebuild=True)
    
    def _initialize_knowledge_base(self, force_rebuild: bool = False):
        """Initialize knowledge base."""
        try:
            if force_rebuild or not self.knowledge_base.is_indexed:
                print("Building fresh knowledge base with updated tables...")
                self.knowledge_base.build_index(self.schema)
        except Exception as e:
            print(f"Failed to build knowledge base: {e}")
    
    def process_query(self, user_question: str) -> Tuple[Optional[pd.DataFrame], str, List[str]]:
        """Process user query using RAG approach with secure execution."""
        
        # Step 1: Retrieve relevant schema knowledge (AI only sees this)
        try:
            relevant_knowledge = self.knowledge_base.retrieve_relevant_knowledge(user_question, top_k=5)
            knowledge_context = "\n".join([item.content for item in relevant_knowledge])
        except Exception as e:
            knowledge_context = f"Error retrieving knowledge: {e}"
        
        # Step 2: AI generates response based on knowledge only (no DB access)
        ai_response = self._generate_ai_response(user_question, knowledge_context)
        
        # Step 3: Try to execute query through secure interface
        try:
            df, query_key, sql = self.secure_interface.execute_query_safely(user_question, self.schema)
            summary = self._format_results_with_ai(df, ai_response, query_key)
            
            # Return knowledge sources for transparency
            sources = [item.metadata for item in relevant_knowledge]
            
            return df, summary, sources
            
        except _UseMistralFallback:
            # Let chat use Mistral to generate SELECT with WHERE
            raise
        except Exception as e:
            # If secure execution fails, return AI analysis only
            return None, f"AI Analysis: {ai_response}\n\nNote: Could not execute query - {e}", []
    
    def _generate_ai_response(self, question: str, knowledge_context: str) -> str:
        """Generate AI response based only on knowledge context."""
        try:
            from mistralai import Mistral
            
            api_key = os.getenv("MISTRAL_API_KEY")
            if not api_key:
                return "No AI API key available. Based on schema knowledge, this appears to be a database query request."
            
            client = Mistral(api_key=api_key)
            model = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
            
            prompt = f"""Based on the following database schema knowledge, provide a helpful response to the user's question.
            
Schema Knowledge:
{knowledge_context}

User Question: {question}

Provide a helpful response about what the user is asking for. If this requires a database query, explain what information would be retrieved.
Do NOT generate SQL queries - only explain what the query would accomplish."""
            
            response = client.chat.complete(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"AI analysis unavailable: {e}"
    
    def _format_results_with_ai(self, df: pd.DataFrame, ai_response: str, query_key: str) -> str:
        """Format query results with AI context."""
        if df.empty:
            return f"Used predefined query: {query_key}. {ai_response}\n\nResult: No data found."
        
        row_count = len(df)
        col_count = len(df.columns)
        
        result = f"Used predefined query: {query_key}. {ai_response}\n\n"
        result += f"Query returned {row_count} row(s) with {col_count} column(s)."
        
        if row_count > 0 and row_count <= 5:
            result += f"\n\nData preview:\n{df.to_string(index=False)}"
        elif row_count > 5:
            result += f"\n\nFirst few rows:\n{df.head(3).to_string(index=False)}"
        
        return result


# Interface for integration with existing chat system
class RAGChatInterface:
    """Interface for RAG system that integrates with existing chat."""
    
    def __init__(self, schema: str = "public"):
        self.assistant = RAGDatabaseAssistant(schema)
    
    def ask_database(self, question: str, schema: str = "public") -> Tuple[Optional[pd.DataFrame], str]:
        """Interface method compatible with existing chat system."""
        df, response, sources = self.assistant.process_query(question)
        
        # Add source information to response
        if sources:
            source_info = "\n\nKnowledge sources used: " + ", ".join([str(s.get('type', 'unknown')) for s in sources])
            response += source_info
        
        return df, response
