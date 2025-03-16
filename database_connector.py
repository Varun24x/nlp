        # Using PyMySQL for MySQL connection via SQLAlchemy
from sqlalchemy import create_engine
import pymysql

class DatabaseConnector:
    def __init__(self, host, user, password, database):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.engine = self.create_engine()

    def create_engine(self):
        # Using PyMySQL for MySQL connection via SQLAlchemy
        db_url = f"mysql+pymysql://{self.user}:{self.password}@{self.host}/{self.database}"
        return create_engine(db_url)

    def get_connection(self):
        return self.engine.connect()
