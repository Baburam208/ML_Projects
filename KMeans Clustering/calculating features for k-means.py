import os
import pandas as pd
import numpy as np
from scipy.stats import gmean


# Function to calculate annual return and volatility
def compute_annual_metrics(df, column_name='close', trading_days=230):
    """
    Compute annual return and volatility for a stock data DataFrame.

    Parameters:
        df (pd.DataFrame): The stock data containing a column for close prices.
        column_name (str): The column containing closing prices.
        trading_days (int): Number of trading days in a year
                            (default: 230, because, in our case 230 is avg trading days).

    Returns:
        tuple: (annual_return, annual_volatility)
    """
    # Calculate daily returns
    daily_returns = df[column_name].pct_change().dropna()

    # Annual return
    annual_return = gmean(1 + daily_returns)**trading_days - 1

    # Annual volatility
    annual_volatility = daily_returns.std() * np.sqrt(trading_days)

    return annual_return, annual_volatility


# Function to process CSV files in a directory structure
def process_directory(base_dir, output_file, column_name='close'):
    """
    Process all CSV files in a directory structure and compute metrics.

    Parameters:
        base_dir (str): Base directory containing subdirectories with CSV files.
        output_file (str): File path to save the resulting CSV.
        column_name (str): The column containing closing prices (default: 'close').

    Returns:
        None
    """
    results = []
    invalid_files = []  # To store files with NaN or infinite values

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                stock_name = os.path.splitext(file)[0]

                # Read CSV file
                df = pd.read_csv(file_path)

                print(f"Checking file: {file_path}")
                print(f"NaN values in 'close': {df[column_name].isna().sum()}")
                print(f"Infinite values in 'close': {np.isinf(df[column_name]).sum()}")

                # Check for NaN or infinite values in the 'close' column
                if df[column_name].isna().any() or np.isinf(df[column_name]).any():
                    invalid_files.append(file)  # Log invalid file
                    continue  # Skip processing this file

                # Compute metrics
                try:
                    annual_return, annual_volatility = compute_annual_metrics(df, column_name=column_name)
                    results.append({
                        'stock': stock_name,
                        'annual return': annual_return,
                        'annual volatility': annual_volatility
                    })
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    # Save results to a new CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    # Log invalid files
    if invalid_files:
        print("Files with NaN or infinite values in the 'close' column:")
        for invalid_file in invalid_files:
            print(invalid_file)


if __name__ == "__main__":
    base_dir = "./Final Fundamental Imputed Data"
    output_file = "./annual_return_and_volatility_metrics.csv"
    process_directory(base_dir, output_file)
