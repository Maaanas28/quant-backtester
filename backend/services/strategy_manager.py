import json
from typing import List, Dict

from database.database import DATABASE_URL, get_db

class StrategyManager:
    """Strategy management for trading strategies"""
    
    def __init__(self):
        """Initialize the strategy manager"""
        pass

    def create_strategy(self, conn, name: str, strategy_type: str, parameters: Dict) -> Dict:
        """Create a new trading strategy"""
        from database.database import USE_POSTGRES
        cursor = conn.cursor()
        try:
            if USE_POSTGRES:
                cursor.execute(
                    "INSERT INTO agents (name, type, parameters) VALUES (%s, %s, %s)",
                    (name, strategy_type, json.dumps(parameters))
                )
            else:
                cursor.execute(
                    "INSERT INTO agents (name, type, parameters) VALUES (?, ?, ?)",
                    (name, strategy_type, json.dumps(parameters))
                )
            conn.commit()
            return {"id": cursor.lastrowid, "name": name, "type": strategy_type, "parameters": parameters}
        except Exception as e:
            # Handle both SQLite IntegrityError and PostgreSQL IntegrityError
            if "unique" in str(e).lower() or "integrity" in str(e).lower():
                raise ValueError(f"Strategy with name '{name}' already exists.")
            raise

    def get_strategies(self, conn) -> List[Dict]:
        """Get all trading strategies"""
        from database.database import USE_POSTGRES
        if USE_POSTGRES:
            from psycopg2.extras import RealDictCursor
            cursor = conn.cursor(cursor_factory=RealDictCursor)
        else:
            cursor = conn.cursor()
        cursor.execute("SELECT id, name, type, parameters FROM agents")
        strategies_data = cursor.fetchall()
        
        strategies = []
        for row in strategies_data:
            strategy = dict(row)
            strategy['parameters'] = json.loads(strategy['parameters'])
            strategies.append(strategy)
        return strategies

    def get_strategy_by_id(self, conn, strategy_id: int) -> Dict:
        """Get a specific strategy by ID"""
        from database.database import USE_POSTGRES
        if USE_POSTGRES:
            from psycopg2.extras import RealDictCursor
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT id, name, type, parameters FROM agents WHERE id = %s", (strategy_id,))
        else:
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, type, parameters FROM agents WHERE id = ?", (strategy_id,))
        strategy_data = cursor.fetchone()
        if strategy_data:
            strategy = dict(strategy_data)
            strategy['parameters'] = json.loads(strategy['parameters'])
            return strategy
        return None
