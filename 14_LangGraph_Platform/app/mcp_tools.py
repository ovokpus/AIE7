"""
MCP Tool Functions

This module contains the actual tool functions that can be used both
by the MCP server and directly in examples/demos.
"""

import os
import csv
import json
import urllib.request
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import statistics


def read_file_content(file_path: str) -> str:
    """
    Read the contents of a file.
    
    Args:
        file_path: Path to the file to read
        
    Returns:
        The contents of the file as a string
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File '{file_path}' not found"
    except PermissionError:
        return f"Error: Permission denied to read '{file_path}'"
    except Exception as e:
        return f"Error reading file: {str(e)}"


def write_file_content(file_path: str, content: str) -> str:
    """
    Write content to a file.
    
    Args:
        file_path: Path to the file to write
        content: Content to write to the file
        
    Returns:
        Success or error message
    """
    try:
        # Create directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote content to '{file_path}'"
    except PermissionError:
        return f"Error: Permission denied to write to '{file_path}'"
    except Exception as e:
        return f"Error writing file: {str(e)}"


def list_directory_contents(directory_path: str = ".") -> str:
    """
    List the contents of a directory.
    
    Args:
        directory_path: Path to the directory to list (defaults to current directory)
        
    Returns:
        JSON string containing directory contents with file/folder info
    """
    try:
        path = Path(directory_path)
        if not path.exists():
            return f"Error: Directory '{directory_path}' does not exist"
        
        if not path.is_dir():
            return f"Error: '{directory_path}' is not a directory"
        
        contents = []
        for item in path.iterdir():
            item_info = {
                "name": item.name,
                "type": "directory" if item.is_dir() else "file",
                "size": item.stat().st_size if item.is_file() else None,
                "modified": datetime.fromtimestamp(item.stat().st_mtime).isoformat()
            }
            contents.append(item_info)
        
        return json.dumps(contents, indent=2)
    except PermissionError:
        return f"Error: Permission denied to access '{directory_path}'"
    except Exception as e:
        return f"Error listing directory: {str(e)}"


def analyze_csv_data(file_path: str, operation: str = "summary") -> str:
    """
    Analyze CSV data with various operations.
    
    Args:
        file_path: Path to the CSV file
        operation: Type of analysis ('summary', 'columns', 'sample', 'stats')
        
    Returns:
        Analysis results as a string
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        if not rows:
            return "Error: CSV file is empty or has no data rows"
        
        if operation == "summary":
            return f"CSV Summary:\n- Total rows: {len(rows)}\n- Columns: {list(rows[0].keys())}\n- Sample row: {rows[0]}"
        
        elif operation == "columns":
            return f"Columns in CSV: {list(rows[0].keys())}"
        
        elif operation == "sample":
            sample_size = min(5, len(rows))
            return f"Sample data (first {sample_size} rows):\n" + json.dumps(rows[:sample_size], indent=2)
        
        elif operation == "stats":
            numeric_columns = []
            for col in rows[0].keys():
                try:
                    # Try to convert first non-empty value to float
                    for row in rows:
                        if row[col] and row[col].strip():
                            float(row[col])
                            numeric_columns.append(col)
                            break
                except (ValueError, TypeError):
                    continue
            
            stats_result = {"numeric_columns": numeric_columns}
            for col in numeric_columns:
                values = []
                for row in rows:
                    try:
                        if row[col] and row[col].strip():
                            values.append(float(row[col]))
                    except (ValueError, TypeError):
                        continue
                
                if values:
                    stats_result[col] = {
                        "count": len(values),
                        "mean": statistics.mean(values),
                        "median": statistics.median(values),
                        "min": min(values),
                        "max": max(values)
                    }
            
            return json.dumps(stats_result, indent=2)
        
        else:
            return f"Error: Unknown operation '{operation}'. Use 'summary', 'columns', 'sample', or 'stats'"
    
    except FileNotFoundError:
        return f"Error: CSV file '{file_path}' not found"
    except Exception as e:
        return f"Error analyzing CSV: {str(e)}"


def get_current_time(timezone: str = "UTC") -> str:
    """
    Get the current date and time.
    
    Args:
        timezone: Timezone to use (currently only supports UTC)
        
    Returns:
        Current date and time as ISO format string
    """
    now = datetime.utcnow()
    return f"Current time (UTC): {now.isoformat()}"


def validate_url(url: str) -> str:
    """
    Validate if a URL is properly formatted and accessible.
    
    Args:
        url: URL to validate
        
    Returns:
        Validation result with status and details
    """
    try:
        # Parse URL
        parsed = urllib.parse.urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return f"Invalid URL format: {url}"
        
        # Try to access URL
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            status_code = response.getcode()
            content_type = response.headers.get('Content-Type', 'Unknown')
            
        return f"URL is valid and accessible:\n- Status: {status_code}\n- Content-Type: {content_type}\n- URL: {url}"
    
    except urllib.error.URLError as e:
        return f"URL not accessible: {url}\nError: {str(e)}"
    except Exception as e:
        return f"Error validating URL: {str(e)}"


def get_environment_info() -> str:
    """
    Get system environment information.
    
    Returns:
        System environment details as JSON string
    """
    try:
        info = {
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "platform": os.name,
            "current_directory": os.getcwd(),
            "environment_variables": {
                key: value for key, value in os.environ.items() 
                if key in ['PATH', 'HOME', 'USER', 'SHELL', 'LANG', 'PWD']
            }
        }
        return json.dumps(info, indent=2)
    except Exception as e:
        return f"Error getting environment info: {str(e)}"


def calculate_statistics(numbers: List[float], stat_type: str = "all") -> str:
    """
    Calculate statistics for a list of numbers.
    
    Args:
        numbers: List of numbers to analyze
        stat_type: Type of statistics ('mean', 'median', 'std', 'all')
        
    Returns:
        Statistical analysis results
    """
    try:
        if not numbers:
            return "Error: No numbers provided"
        
        if stat_type == "mean":
            return f"Mean: {statistics.mean(numbers)}"
        elif stat_type == "median":
            return f"Median: {statistics.median(numbers)}"
        elif stat_type == "std":
            if len(numbers) > 1:
                return f"Standard Deviation: {statistics.stdev(numbers)}"
            else:
                return "Error: Need at least 2 numbers for standard deviation"
        elif stat_type == "all":
            result = {
                "count": len(numbers),
                "mean": statistics.mean(numbers),
                "median": statistics.median(numbers),
                "min": min(numbers),
                "max": max(numbers),
                "sum": sum(numbers)
            }
            if len(numbers) > 1:
                result["standard_deviation"] = statistics.stdev(numbers)
            
            return json.dumps(result, indent=2)
        else:
            return f"Error: Unknown stat_type '{stat_type}'. Use 'mean', 'median', 'std', or 'all'"
    
    except Exception as e:
        return f"Error calculating statistics: {str(e)}"
