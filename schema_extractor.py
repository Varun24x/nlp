from sqlalchemy import create_engine, inspect

class SchemaExtractor:
    def __init__(self, db_connector):
        self.db_connector = db_connector

    def get_schema(self):
        inspector = inspect(self.db_connector.engine)
        schema = {}
        for table_name in inspector.get_table_names():
            columns = inspector.get_columns(table_name)
            schema[table_name] = [column['name'] for column in columns]
        return schema
