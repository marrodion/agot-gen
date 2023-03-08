from pathlib import Path

import pandas as pd
import typer
from sklearn.model_selection import train_test_split


def main(
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


if __name__ == "__main__":
    typer.run(main)
