"""
CSV file writing utility
"""
import csv
import os


def write2file(filename, row):
    """
    Append a row to CSV file
    
    Args:
        filename: Output CSV filename
        row: List of values to write as a row
    """
    with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row)


def refresh_csv(filename):
    """
    Delete existing CSV file if it exists
    
    Args:
        filename: CSV filename to delete
    """
    if os.path.exists(filename):
        os.remove(filename)
        print(f'Refreshed {filename}')
