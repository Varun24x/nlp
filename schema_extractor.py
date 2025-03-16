from sqlalchemy import create_engine, inspect

class SchemaExtractor:
    def __init__(self, db_connector):
        self.db_connector = db_connector

    def get_schema(self):
        """Fetch database schema dynamically for MySQL and PostgreSQL."""
        try:
            inspector = inspect(self.db_connector.engine)
            schema = {}

            # Get table names (handle PostgreSQL schema explicitly)
            if self.db_connector.db_type == "postgresql":
                table_names = inspector.get_table_names(schema="public")
            else:
                table_names = inspector.get_table_names()

            # Fetch column names for each table
            for table_name in table_names:
                columns = inspector.get_columns(table_name)
                schema[table_name] = [column["name"] for column in columns]

            return schema if schema else "No tables found in the database."

        except Exception as e:
            return f"Error extracting schema: {str(e)}"
