"""Explore the 2015–2024 World Happiness dataset with a CodeAgent."""

import os
from pathlib import Path

import matplotlib
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import pearsonr
from smolagents import CodeAgent, OpenAIServerModel, tool

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "happiness_project"
OUTPUT_DIR = ROOT / "outputs"
COLUMN_NAMES = {
    "Country": "country",
    "Regional indicator": "region",
    "Happiness score": "happiness_score",
    "Ladder score": "happiness_score",
    "GDP per capita": "gdp_per_capita",
}
df = None


@tool
def load_happiness_data() -> dict:
    """Load and merge the yearly World Happiness CSVs into memory.

    Returns:
        A dictionary with the merged shape and column names, or an error.
    """
    global df
    frames = []
    for path in sorted(DATA_DIR.glob("world_happiness_*.csv")):
        year_df = pd.read_csv(path, sep=";", decimal=",").rename(columns=COLUMN_NAMES)
        year_df["year"] = int(path.stem.rsplit("_", 1)[1])
        frames.append(year_df)

    if not frames:
        return {"error": "No World Happiness CSVs found."}

    df = pd.concat(frames, ignore_index=True)
    return {"shape": df.shape, "columns": df.columns.tolist()}


@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for a column in the loaded dataset.

    Args:
        column: Name of the column to summarize.

    Returns:
        A dictionary of descriptive statistics, or an error.
    """
    if df is None or column not in df.columns:
        return {"error": "Load the data and choose an existing column."}
    return df[column].describe().to_dict()


@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Calculate Pearson r and its p-value for two numeric columns.

    Args:
        col1: Name of the first numeric column.
        col2: Name of the second numeric column.

    Returns:
        A dictionary with both column names, Pearson r, and p-value, or an error.
    """
    if df is None or col1 not in df.columns or col2 not in df.columns:
        return {"error": "Load the data and choose two existing columns."}

    x = pd.to_numeric(df[col1], errors="coerce")
    y = pd.to_numeric(df[col2], errors="coerce")
    valid = x.notna() & y.notna()
    if valid.sum() < 2 or x[valid].nunique() < 2 or y[valid].nunique() < 2:
        return {"error": "Need at least two paired values and variation in both columns."}

    r, p = pearsonr(x[valid], y[valid])
    return {
        "col1": col1,
        "col2": col2,
        "pearson_r": round(float(r), 4),
        "p_value": round(float(p), 4),
    }


@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> list[dict] | dict:
    """Rank countries by a numeric column in a given year.

    Args:
        column: Name of the numeric column to rank by.
        year: Calendar year to include.
        n: Maximum number of countries to return.

    Returns:
        A list of country and value dictionaries, or an error dictionary.
    """
    if df is None or column not in df.columns:
        return {"error": "Load the data and choose an existing column."}
    if n < 1 or not pd.api.types.is_numeric_dtype(df[column]):
        return {"error": "Use a positive count and a numeric ranking column."}
    rows = df.loc[df["year"] == year, ["country", column]].dropna()
    if rows.empty:
        return {"error": f"No ranked data available for {year}."}
    return rows.sort_values(column, ascending=False).head(n).to_dict("records")


SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the provided tools for loading, summarizing, correlating, and ranking.
The load_happiness_data tool returns a dictionary of shape and column names.
Write Python for analyses the tools do not cover. When supplied, happiness_data
is the complete normalized pandas DataFrame. Use its actual rows for code-based
analyses. Regional trends should use mean happiness_score per region and year.
Save plots in the supplied plot_path and close figures afterward.
A p-value returned as zero was rounded to four decimals. Report it as very
small and explain statistical significance without claiming causation.
"""

GUIDED_QUERIES = [
    "Load the happiness data and tell me its shape and column names.",
    "Summarize the happiness_score column.",
    "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
    "Show me the top 5 happiest countries in 2020.",
    "Use happiness_data to group happiness_score by year and region and compute the mean. "
    "Plot 2015–2024 with one line per region using the actual region names in the legend. "
    "Save the chart to plot_path. Use the provided DataFrame directly.",
]


def main():
    matplotlib.use("Agg")
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY in your course .env before running Week 7.")

    OUTPUT_DIR.mkdir(exist_ok=True)
    agent = CodeAgent(
        tools=[load_happiness_data, summarize_column, compute_correlation, get_top_n_countries],
        model=OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini"),
        instructions=SYSTEM_PROMPT,
        additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats", "pathlib"],
        max_steps=8,
    )
    for query in GUIDED_QUERIES[:-1]:
        print(f"\n--- Query: {query} ---")
        print(agent.run(query, reset=False))

    chart = OUTPUT_DIR / "happiness_by_region.png"
    if df is None:
        raise RuntimeError("The agent did not load the happiness dataset.")
    previous_mtime = chart.stat().st_mtime_ns if chart.is_file() else None
    data_args = {"happiness_data": df.copy(), "plot_path": str(chart)}
    query = GUIDED_QUERIES[-1]
    print(f"\n--- Query: {query} ---")
    print(agent.run(query, reset=False, additional_args=data_args))
    if not chart.is_file() or chart.stat().st_mtime_ns == previous_mtime:
        print(f"Check the agent's plot step. The expected chart is missing at {chart}.")

    my_query_1 = "Which five countries ranked highest on GDP per capita in 2024?"
    print(f"\n--- Query: {my_query_1} ---")
    print(agent.run(my_query_1, reset=False))
    # The agent called get_top_n_countries for 2024 GDP per capita.
    # Its tool result ranked Luxembourg first, followed by Ireland.

    my_query_2 = (
        "Which region had the largest absolute change in mean happiness from 2015 to 2024? "
        "Report both yearly means and the signed change."
    )
    print(f"\n--- Query: {my_query_2} ---")
    print(agent.run(my_query_2, reset=False, additional_args=data_args))
    # The agent wrote pandas code to group by region and compare the two years.
    # South Asia had the largest absolute change: 4.5800 to 3.8957 (-0.6843).


if __name__ == "__main__":
    main()

# --- Reflection ---
# 1. The agent reported Pearson r = 0.6313 and a very small p-value.
#    This is a moderately strong positive relationship and is statistically
#    significant under the test assumptions. The tool rounded p to 0.0000.
#    The agent could have explained the direction and significance more clearly.
# 2. The first plot contained only one point even though the agent said it
#    had finished. Inspecting the PNG exposed the error. Giving the agent the
#    cleaned DataFrame produced the expected ten regional lines on the rerun.
# 3. A regional trend tool could return yearly means and country counts per
#    region. It would help answer whether an apparent change coincided with
#    changing country coverage.
