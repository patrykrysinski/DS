from langchain.tools import BaseTool, tool
import pandas as pd
import matplotlib.pyplot as plt
import io


def load_csv(file_path: str) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: 'data' or error message as 'error'
    """
    return pd.read_csv(file_path)

@tool
def describe_dataframe(file_path: str) -> str:
    """Provide a description of the DataFrame structure.
    
    Args:
        df_input (dict): String to the file location 
        
    Returns:
        str: String containing description or error message
    """
    try:
        df = pd.read_csv(file_path)
        buffer = io.StringIO()
        df.info(buf=buffer)
        description = buffer.getvalue()
        return f"DataFrame Description:\n{description}"
    except Exception as e:
        return f"Error describing DataFrame: {e}"

# @tool
# def summarize_dataframe(df_input: dict) -> dict:
#     """Summarize the contents of a pandas DataFrame.
    
#     Args:
#         df_input (dict): Dictionary containing DataFrame from load_csv tool
        
#     Returns:
#         dict: Dictionary containing summary statistics or error message
#     """
#     try:
#         if not isinstance(df_input, dict) or 'data' not in df_input:
#             df = df_input if isinstance(df_input, pd.DataFrame) else None
#         else:
#             df = df_input['data']
            
#         if df is None or not isinstance(df, pd.DataFrame):
#             return {
#                 'success': False,
#                 'error': 'Invalid input: Expected DataFrame or dict with DataFrame'
#             }
            
#         summary = df.describe().to_string()
#         return {
#             'success': True,
#             'data': summary,
#             'message': 'Successfully generated summary statistics'
#         }
#     except Exception as e:
#         return {
#             'success': False,
#             'error': f"Error summarizing DataFrame: {e}"
#         }

@tool 
def columns_in_df(file_path: str) -> str:
    """Return the column names of the DataFrame.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        str: Comma-separated column names or error message
    """
    try:
        df = pd.read_csv(file_path)
        columns = df.columns.tolist()
        return f"Columns in DataFrame: {', '.join(columns)}"
    except Exception as e:
        return f"Error retrieving columns: {e}"

@tool
def summarize_expenses_by_category(
    file_path: str, amount_column: str, category_column: str
) -> str:
    """Summarize expenses by category from a pandas DataFrame.
    
    Args:
        file_path (str): Path to the CSV file
        amount_column (str): Name of the column containing amounts
        category_column (str): Name of the column containing categories
        
    Returns:
        str: Summary of expenses by category or error message
    """
    try:
        df = pd.read_csv(file_path)
        summary = df.groupby(category_column)[amount_column].sum().to_string()
        return f"Expense Summary by Category:\n{summary}"
    except Exception as e:
        return f"Error summarizing expenses: {e}"
# @tool
# def plot_expenses(
#     df_input: dict, amount_column: str, category_column: str, output_file: str
# ) -> dict:
#     """Plot expenses by category from a pandas DataFrame and save to a file.
    
#     Args:
#         df_input (dict): Dictionary containing DataFrame from load_csv tool
#         amount_column (str): Name of the column containing amounts
#         category_column (str): Name of the column containing categories
#         output_file (str): Path where to save the plot
        
#     Returns:
#         dict: Dictionary containing success status and file path or error message
#     """
#     try:
#         if not isinstance(df_input, dict) or 'data' not in df_input:
#             df = df_input if isinstance(df_input, pd.DataFrame) else None
#         else:
#             df = df_input['data']
            
#         if df is None or not isinstance(df, pd.DataFrame):
#             return {
#                 'success': False,
#                 'error': 'Invalid input: Expected DataFrame or dict with DataFrame'
#             }

#         summary = df.groupby(category_column)[amount_column].sum()
#         plt.figure(figsize=(10, 6))
#         summary.plot(kind='bar')
#         plt.xlabel('Category')
#         plt.ylabel('Total Expenses')
#         plt.title('Expenses by Category')
#         plt.tight_layout()
#         plt.savefig(output_file)
#         plt.close()  # Clean up the figure
        
#         return {
#             'success': True,
#             'data': output_file,
#             'message': f"Expense plot saved to {output_file}"
#         }
#     except Exception as e:
#         return {
#             'success': False,
#             'error': f"Error plotting expenses: {e}"
#         }
