# %% Define function to convert DataFrame to markdown table string:
from pandas import DataFrame


def table_to_markdown(df: DataFrame) -> str:
    """
    Convert a pandas DataFrame to a markdown table string using tabulate
    """

    # If df is empty, return an empty string
    if df.empty:
        return "(empty table)"

    # If df is a series, convert it to a DataFrame
    if isinstance(df, DataFrame) is False:
        df = DataFrame(df)

    try:
        from tabulate import tabulate
        return tabulate(df, headers='keys', tablefmt='github', showindex=True)
    except Exception as e:
        #Print stack trace for debugging
        import traceback
        traceback.print_exc()
        return df.to_string(index=True, justify='left')


def quote_text(text: str) -> str:
    """
    Quote text for markdown formatting.

    Parameters
    ----------
    text : str
        The text to quote.

    Returns
    -------
    str
        The quoted text.
    """
    newline_with_quote = '\n> '
    quoted_text = text.replace('\n', newline_with_quote)
    return f"> {quoted_text}"