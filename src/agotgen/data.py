import csv
from pathlib import Path
from typing import List

import pandas as pd
import typer
from sklearn.model_selection import train_test_split

app = typer.Typer()


@app.command()
def preprocess_kenlm(filename: Path, key: List[str]):
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


@app.command()
def split(
    seed: int = typer.Option(...),
    test_size: float = typer.Option(...),
    filename: Path = typer.Option(...),
):
    key = ["type_code", "faction_code", "is_unique"]
    df = pd.read_json(filename, orient="records", lines=True)
    key = df[key].astype(str).sum(axis=1)
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=key,
    )
    train.to_json("./data/splits/train.jsonl", orient="records", lines=True)
    test.to_json("./data/splits/test.jsonl", orient="records", lines=True)


def cleanup(text: pd.Series) -> pd.Series:
    text = text.str.lower()
    text = text.str.replace("\n", "", regex=False)
    text = text.str.replace(r"\.\d", "", regex=True)
    text = text.str.replace(r"\.\S", ". ", regex=True)
    text = text.str.replace(r"\s+", " ", regex=True)
    return text
