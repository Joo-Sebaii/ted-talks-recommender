"""
Utility functions for the TED Talks Recommendation System
"""

def validate_query(query):
    """
    Ensure the user query is valid.
    """
    if not isinstance(query, list):
        raise ValueError("Query must be a list of strings")

    if len(query) == 0:
        raise ValueError("Query list cannot be empty")

    return True


def print_header(title):
    """
    Pretty print section headers in console output.
    """
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50 + "\n")
