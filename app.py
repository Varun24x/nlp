import streamlit as st
from database_connector import DatabaseConnector
from query_executor import QueryExecutor
from schema_extractor import SchemaExtractor
from spell_corrector import SpellCorrector
import logging
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(
    filename='logs/app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def save_feedback(nl_query, generated_sql, corrected_sql, success):
    """Save user feedback to CSV."""
    feedback_data = {
        'timestamp': datetime.now(),
        'nl_query': nl_query,
        'generated_sql': generated_sql,
        'corrected_sql': corrected_sql,
        'success': success
    }
    df = pd.DataFrame([feedback_data])
    feedback_path = 'logs/user_feedback.csv'

    # Append feedback to CSV
    df.to_csv(feedback_path, mode='a', header=not pd.io.common.file_exists(feedback_path), index=False)
    logging.info("Feedback saved successfully.")

def main():
    st.title("Natural Language to SQL Converter")

    # Database connection configuration
    with st.sidebar:
        st.header("Database Connection")
        host = st.text_input("Host", "localhost")
        user = st.text_input("Username")
        password = st.text_input("Password", type="password")
        database = st.text_input("Database Name")

    try:
        # Create a db_config dictionary
        db_config = {
            "host": host,
            "user": user,
            "password": password,
            "database": database,
        }

        # Initialize components
        db_connector = DatabaseConnector(host, user, password, database)
        schema_extractor = SchemaExtractor(db_connector)
        spell_corrector = SpellCorrector()
        query_executor = QueryExecutor(model_paths={
                                       "pretrained": "t5-small",  # Path to the pre-trained model
                                       "finetuned": "models/t5_spider_finetuned"  # Path to the fine-tuned model
                                        }, db_config=db_config)

        # Display schema in the sidebar
        schema = schema_extractor.get_schema()
        st.sidebar.header("Database Schema")
        for table, columns in schema.items():
            st.sidebar.subheader(table)
            for column in columns:
                st.sidebar.write(f"- {column}")

        # Query input section
        st.header("Enter your question")
        nl_query = st.text_area("Type your question in natural language")

        # Generate SQL query
        if st.button("Generate SQL"):
            if nl_query.strip():
                try:
                    # Correct spelling
                    corrected_query = spell_corrector.correct(nl_query)

                    # Display corrected NL query
                    st.subheader("Corrected Natural Language Query")
                    st.write(corrected_query)  # Display corrected query

                    # Generate SQL query using the corrected input
                    sql_query = query_executor.generate_query(corrected_query)

                    # Store SQL query in session state
                    st.session_state['sql_query'] = sql_query

                    # Display generated SQL
                    st.subheader("Generated SQL Query")
                    st.code(sql_query, language="sql")

                except Exception as e:
                    st.error(f"Error generating SQL: {str(e)}")
                    logging.error(f"Error generating SQL: {str(e)}")

        # Execute SQL query and display results
        if "sql_query" in st.session_state and st.button("Execute SQL"):
            try:
                sql_query = st.session_state['sql_query']

                # Get the database connection
                connection = query_executor.connect_to_database()
                if not connection:
                    st.error("Failed to connect to the database.")
                    return

                # Display SQL query before execution
                st.subheader("Executing SQL Query")
                st.code(sql_query, language="sql")

                # Execute the SQL query
                success, results = query_executor.execute_query(connection, sql_query)

                # Display results
                st.subheader("Query Results")
                if not success or not results:
                    st.warning("No results returned for the query.")
                else:
                    st.dataframe(results)

            except Exception as e:
                st.error(f"Error executing query: {str(e)}")
                logging.error(f"Error executing query: {str(e)}")

        # Feedback section
        if "sql_query" in st.session_state:
            st.subheader("Feedback")
            correct = st.radio("Is this SQL query correct?", ["Yes", "No"], key="feedback_radio")

            if correct == "No":
                corrected_sql = st.text_area("Please provide the correct SQL query", key="corrected_sql_input")
                if st.button("Submit Feedback"):
                    save_feedback(nl_query, st.session_state['sql_query'], corrected_sql, False)
                    st.success("Thank you for your feedback!")
            else:
                if st.button("Submit Feedback"):
                    save_feedback(nl_query, st.session_state['sql_query'], st.session_state['sql_query'], True)
                    st.success("Thank you for your feedback!")

    except Exception as e:
        st.error(f"Database connection error: {str(e)}")
        logging.error(f"Database connection error: {str(e)}")

if __name__ == "__main__":
    main()
