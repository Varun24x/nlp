from sqlalchemy import create_engine
import logging

class DatabaseConnector:
    def __init__(self, db_type, host, user, password, database):
        """
        Initialize database connection parameters.
        
        Args:
            db_type (str): Database type ("mysql" or "postgresql").
            host (str): Database host.
            user (str): Database username.
            password (str): Database password.
            database (str): Database name.
        """
        self.db_type = db_type.lower()
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.engine = self.create_engine()

    def create_engine(self):
        """Create a database engine based on db_type."""
        try:
            if self.db_type == "mysql":
                db_url = f"mysql+pymysql://{self.user}:{self.password}@{self.host}/{self.database}"
            elif self.db_type == "postgresql":
                db_url = f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}/{self.database}"
            else:
                raise ValueError("Unsupported database type. Use 'mysql' or 'postgresql'.")
            
            engine = create_engine(db_url)
            logging.info(f"Successfully created engine for {self.db_type}.")
            return engine
        except Exception as e:
            logging.error(f"Error creating engine: {e}")
            raise

    def execute_query(self, query):
        """Execute a query and return the results."""
        try:
            with self.engine.connect() as connection:
                result = connection.execute(query)
                columns = result.keys()
                rows = result.fetchall()
                return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logging.error(f"Error executing query: {e}")
            raise

    def close(self):
        """Dispose of the engine to close the connection."""
        if self.engine:
            self.engine.dispose()
            logging.info("Database connection closed.")
