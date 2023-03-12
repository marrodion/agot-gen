import typer
import kenlm
import json

app = typer.Typer()


@app.command()
def evaluate():
    model = kenlm.Model("./models/kenlm/lm.arpa")
    train_perplexity = get_perplexity(model, "./data/lm/train.txt")
    test_perplexity = get_perplexity(model, "./data/lm/test.txt")
    with open("./metrics/kenlm.json", "wt+") as fh:
        json.dump(
            {"perplexity": {"train": train_perplexity, "test": test_perplexity}},
            fh,
        )


def get_perplexity(model, fn) -> float:
    with open(fn) as fh:
        n = 0
        perplexity = 0
        for line in fh:
            perplexity += model.perplexity(line.strip())
            n += 1
    return perplexity / n


if __name__ == "__main__":
    app()
