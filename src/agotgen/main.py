import typer

from agotgen import data
from agotgen.model import kenlm

app = typer.Typer()

app.add_typer(data.app, name="data")
app.add_typer(kenlm.app, name="kenlm")

if __name__ == "__main__":
    app()
