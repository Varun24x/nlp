import logging
import mysql.connector
from mysql.connector import Error
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import Tuple, Dict, Optional

class QueryExecutor:
    def __init__(self, model_paths: Dict[str, str], db_config: Dict):
        """
        Initialize the QueryExecutor with models and database configuration.

        :param model_paths: Dictionary with paths for fine-tuned and pre-trained models.
        :param db_config: Database connection configuration dictionary.
        """
        # Load tokenizer and models
        self.tokenizer = T5Tokenizer.from_pretrained(model_paths["pretrained"])
        self.finetuned_model = T5ForConditionalGeneration.from_pretrained(model_paths["finetuned"])
        self.pretrained_model = T5ForConditionalGeneration.from_pretrained(model_paths["pretrained"])
        self.db_config = db_config

    def connect_to_database(self) -> Optional[mysql.connector.connection.MySQLConnection]:
        """Connect to the MySQL database."""
        try:
            logging.info("Connecting to the database...")
            connection = mysql.connector.connect(**self.db_config)
            if connection.is_connected():
                logging.info("Successfully connected to the database.")
                return connection
        except Error as e:
            logging.error(f"Error while connecting to the database: {e}")
        return None

    def generate_sql_using_rules(self, natural_language_query: str, schema: Optional[Dict[str, list]] = None) -> Optional[str]:
        """
        Generate SQL query using logic rules if the query matches predefined patterns.
        """
        # Rule 1: Get all students
        if "all students" in natural_language_query:
            return "SELECT * FROM Students;"
        
        # Rule 2: Get students by gender
        if "students" in natural_language_query and "male" in natural_language_query:
            return "SELECT * FROM Students WHERE gender = 'Male';"
        if "students" in natural_language_query and "female" in natural_language_query:
            return "SELECT * FROM Students WHERE gender = 'Female';"
        
        # Rule 3: Get attendance for a specific student
        if "attendance" in natural_language_query and "student" in natural_language_query:
            student_name = self.extract_student_name(natural_language_query)
            return f"SELECT date, status FROM Attendance WHERE student_id = (SELECT student_id FROM Students WHERE name = '{student_name}');"
        
        # Rule 4: Get marks for a student in a specific subject
        if "marks" in natural_language_query and "student" in natural_language_query:
            student_name = self.extract_student_name(natural_language_query)
            subject = self.extract_subject(natural_language_query)
            return f"SELECT marks FROM Marks m JOIN Subjects s ON m.subject_id = s.subject_id WHERE student_id = (SELECT student_id FROM Students WHERE name = '{student_name}') AND subject_name = '{subject}';"
        
        # Rule 5: Get students with marks above a threshold in a specific subject
        if "marks" in natural_language_query and "above" in natural_language_query:
            subject = self.extract_subject(natural_language_query)
            threshold = self.extract_marks_threshold(natural_language_query)
            return f"SELECT s.name FROM Marks m JOIN Students s ON m.student_id = s.student_id JOIN Subjects sub ON m.subject_id = sub.subject_id WHERE sub.subject_name = '{subject}' AND m.marks > {threshold};"
        
        # Rule 6: Get attendance count for each student
        if "attendance" in natural_language_query and "count" in natural_language_query:
            return "SELECT s.name, COUNT(a.status) AS attendance_count FROM Attendance a JOIN Students s ON a.student_id = s.student_id WHERE a.status = 'P' GROUP BY s.student_id;"
        
        # If no rules matched, return None to fallback to the T5 model
        return None

    def extract_student_name(self, query: str) -> str:
        # Implement logic to extract student name from the query
        return "John"  # Placeholder implementation

    def extract_subject(self, query: str) -> str:
        # Implement logic to extract subject name from the query
        return "Mathematics"  # Placeholder implementation

    def extract_marks_threshold(self, query: str) -> int:
        # Implement logic to extract marks threshold from the query
        return 80  # Placeholder implementation

    def generate_query(self, natural_language_query: str, schema: Optional[Dict[str, list]] = None, use_finetuned=True) -> str:
        """
        Generate SQL query from natural language input.

        :param natural_language_query: The input query in natural language.
        :param schema: Database schema (table and column names) for better context.
        :param use_finetuned: Boolean flag to toggle between fine-tuned and pre-trained models.
        :return: Generated SQL query or an empty string if failed.
        """
        try:
            logging.info(f"Generating SQL query for: '{natural_language_query}'")
            
            # Try generating the query using logical rules first
            sql_query = self.generate_sql_using_rules(natural_language_query, schema)
            
            # If rules fail, fall back to T5 model
            if sql_query is None:
                logging.info("No rule matched. Using T5 model to generate SQL query.")
                model = self.finetuned_model if use_finetuned else self.pretrained_model

                # Prepare input text
                schema_context = ""
                if schema:
                    schema_context = " | ".join(
                        [f"table {table}: {', '.join(columns)}" for table, columns in schema.items()]
                    )
                input_text = f"translate English to SQL: {natural_language_query}. Schema: {schema_context}"

                # Tokenize and generate query
                inputs = self.tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
                outputs = model.generate(inputs, max_length=512, num_beams=5, early_stopping=True, no_repeat_ngram_size=2)
                sql_query = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                logging.info(f"Generated SQL query: {sql_query}")
            
            return sql_query
        except Exception as e:
            logging.error(f"Error generating SQL query: {e}")
            return ""

    def validate_query(self, connection, sql_query: str) -> bool:
        """Validate the generated SQL query by executing EXPLAIN."""
        try:
            if not sql_query.strip():
                logging.warning("Generated SQL query is empty.")
                return False
            cursor = connection.cursor()
            cursor.execute(f"EXPLAIN {sql_query}")
            logging.info("SQL query validated successfully.")
            return True
        except Error as e:
            logging.warning(f"Query validation failed: {e}")
            return False

    def execute_query(self, connection, sql_query: str) -> Tuple[bool, list]:
        """Execute the SQL query and fetch results."""
        try:
            logging.debug(f"Executing SQL query: {sql_query}")  # Debug log to verify sql_query
            cursor = connection.cursor()
            cursor.execute(sql_query)  # Ensure sql_query is passed here
            results = cursor.fetchall()
            logging.info("SQL query executed successfully.")
            return True, results
        except Error as e:
            logging.error(f"Error executing SQL query: {e}")
            return False, []

    def process_query(self, natural_language_query: str, schema: Optional[Dict[str, list]] = None) -> Tuple[bool, list]:
        """
        Process a natural language query to generate, validate, and execute an SQL query.

        :param natural_language_query: The input natural language query.
        :param schema: Database schema for better query generation.
        :return: A tuple containing a success flag and the result set.
        """
        connection = self.connect_to_database()
        if not connection:
            return False, []

        try:
            # Step 1: Generate SQL Query using fine-tuned model
            sql_query = self.generate_query(natural_language_query, schema=schema, use_finetuned=True)
            logging.debug(f"Generated SQL query (fine-tuned): {sql_query}")  # Debug log to track generated query

            # Fallback to pre-trained model if validation fails
            if not self.validate_query(connection, sql_query):
                logging.warning("Generated query failed validation. Falling back to pre-trained model.")
                sql_query = self.generate_query(natural_language_query, schema=schema, use_finetuned=False)
                logging.debug(f"Generated SQL query (pre-trained): {sql_query}")  # Debug log to track fallback query
                if not self.validate_query(connection, sql_query):
                    logging.error("Query validation failed for both models.")
                    return False, []

            # Step 2: Execute SQL Query
            if sql_query.strip() == "":
                logging.error("Generated SQL query is empty. Cannot execute.")
                return False, []
                
            logging.debug(f"About to execute SQL query: {sql_query}")  # Debug log before executing
            success, results = self.execute_query(connection, sql_query)  # Correctly pass sql_query here
            return success, results
        finally:
            if connection.is_connected():
                connection.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)  # Change to DEBUG to capture more detailed logs

    # Database configuration
    db_config = {
        "host": "localhost",
        "user": "your_username",
        "password": "your_password",
        "database": "your_database",
    }

    # Model paths
    model_paths = {
        "pretrained": "t5-small",  # Replace with actual path to the pretrained model
        "finetuned": "models/t5_spider_finetuned",  # Replace with the path to your fine-tuned model
    }

    # Instantiate QueryExecutor
    executor = QueryExecutor(model_paths=model_paths, db_config=db_config)

    # Example query and schema
    nl_query = "What are the names of employees in the IT department?"
    schema = {"employees": ["id", "name", "department", "salary"]}

    # Process the query and print results
    success, results = executor.process_query(nl_query, schema=schema)
    if success:
        print("Query executed successfully. Results:")
        for row in results:
            print(row)
    else:
        print("Failed to execute the query.")
