"""
Database module supporting both SQLite (local) and PostgreSQL (production)
"""
import sqlite3
import json
import os
from config import config

# Determine database type and connection
DATABASE_URL = config.database.URL if hasattr(config.database, 'URL') else "./database/neuroquant.db"
DATABASE_TYPE = os.getenv("DATABASE_TYPE", "sqlite")  # sqlite or postgresql

# Check if PostgreSQL URL is provided (Render auto-provides this)
if DATABASE_URL.startswith("postgresql://") or DATABASE_URL.startswith("postgres://"):
    DATABASE_TYPE = "postgresql"
    # Import PostgreSQL dependencies only if needed
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        USE_POSTGRES = True
    except ImportError:
        print("WARNING: psycopg2 not installed. Install with: pip install psycopg2-binary")
        print("Falling back to SQLite")
        DATABASE_TYPE = "sqlite"
        DATABASE_URL = "./database/neuroquant.db"
        USE_POSTGRES = False
else:
    USE_POSTGRES = False


def get_connection():
    """Get database connection based on DATABASE_TYPE"""
    if USE_POSTGRES:
        # PostgreSQL connection with RealDictCursor for dict-like results
        conn = psycopg2.connect(DATABASE_URL)
        # Note: RealDictCursor is set per-cursor, not per-connection
        return conn
    else:
        # SQLite connection
        conn = sqlite3.connect(DATABASE_URL, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn


def create_db_and_tables():
    """Create tables - works for both SQLite and PostgreSQL"""
    conn = get_connection()
    cursor = conn.cursor()
    
    if USE_POSTGRES:
        # PostgreSQL syntax
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agents (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                type TEXT NOT NULL,
                parameters TEXT NOT NULL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backtest_runs (
                id SERIAL PRIMARY KEY,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                agent_id INTEGER,
                agent_name TEXT,
                test_period TEXT NOT NULL,
                agent_return REAL NOT NULL,
                buy_hold_return REAL NOT NULL,
                outperformance REAL NOT NULL,
                total_trades INTEGER NOT NULL,
                final_value REAL NOT NULL,
                trades TEXT,
                portfolio_history TEXT,
                portfolio_dates TEXT,
                FOREIGN KEY (agent_id) REFERENCES agents(id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS custom_datasets (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                data TEXT NOT NULL
            )
        """)
    else:
        # SQLite syntax
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                type TEXT NOT NULL,
                parameters TEXT NOT NULL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backtest_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                agent_id INTEGER,
                agent_name TEXT,
                test_period TEXT NOT NULL,
                agent_return REAL NOT NULL,
                buy_hold_return REAL NOT NULL,
                outperformance REAL NOT NULL,
                total_trades INTEGER NOT NULL,
                final_value REAL NOT NULL,
                trades TEXT,
                portfolio_history TEXT,
                portfolio_dates TEXT,
                FOREIGN KEY (agent_id) REFERENCES agents(id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS custom_datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                data TEXT NOT NULL
            )
        """)
    
    conn.commit()

    # Add default strategies if they don't exist
    default_strategies = [
        {"name": "MA Crossover (20/50)", "type": "ma_cross", "parameters": {"short_window": 20, "long_window": 50}},
        {"name": "RSI Mean Reversion", "type": "rsi", "parameters": {"period": 14, "oversold": 30, "overbought": 70}},
        {"name": "Momentum (20 days)", "type": "momentum", "parameters": {"lookback": 20}},
        {"name": "Buy & Hold Benchmark", "type": "buy_hold", "parameters": {}}
    ]

    for strategy_data in default_strategies:
        try:
            if USE_POSTGRES:
                cursor.execute(
                    "INSERT INTO agents (name, type, parameters) VALUES (%s, %s, %s) ON CONFLICT (name) DO NOTHING",
                    (strategy_data["name"], strategy_data["type"], json.dumps(strategy_data["parameters"]))
                )
            else:
                cursor.execute(
                    "INSERT OR IGNORE INTO agents (name, type, parameters) VALUES (?, ?, ?)",
                    (strategy_data["name"], strategy_data["type"], json.dumps(strategy_data["parameters"]))
                )
            conn.commit()
            print(f"Default strategy '{strategy_data['name']}' added.")
        except Exception as e:
            print(f"Error adding default strategy '{strategy_data['name']}': {e}")
    
    conn.close()
    print(f"✅ Database initialized successfully using {DATABASE_TYPE.upper()}")


def get_db():
    """Generator for database connections"""
    conn = get_connection()
    if not USE_POSTGRES:
        conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


# Initialize database on module import
create_db_and_tables()

