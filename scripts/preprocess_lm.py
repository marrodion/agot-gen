from pathlib import Path
from typing import List
import typer
import pandas as pd
import csv


def main(filename: Path, key: List[str]):
    df = pd.read_json(filename, orient="records", lines=True)
    text = df[key[0]]
    for col in key[1:]:
        text = text.str.cat(df[col].astype(str), sep=" ")
    text = cleanup(text)
    text.to_csv(
        f"data/lm/{filename.stem}.txt",
        index=False,
        header=False,
        quoting=csv.QUOTE_NONE,
        escapechar="",
        sep="|",
        quotechar="",
    )


def cleanup(text: pd.Series) -> pd.Series:
    text = text.str.lower()
    text = text.str.replace("\n", "", regex=False)
    text = text.str.replace(r"\.\d", "", regex=True)
    text = text.str.replace(r"\.\S", ". ", regex=True)
    text = text.str.replace(r"\s+", " ", regex=True)
    return text


if __name__ == "__main__":
    typer.run(main)
