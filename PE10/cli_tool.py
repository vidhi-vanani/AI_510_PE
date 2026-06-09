import click
import joblib
import logging
import os
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

logging.basicConfig(
    filename="mlops.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


@click.group()
def cli():
    """MLOps CLI Tool"""
    pass

@cli.command()
def train():
    """Train and save the Iris model."""

    logging.info("Training started")

    click.echo("Training model...")

    X, y = load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LogisticRegression(max_iter=200)

    model.fit(X_train, y_train)

    os.makedirs("model", exist_ok=True)

    joblib.dump(model, "model/trained_model.pkl")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    click.echo(f"Model trained with accuracy: {accuracy:.2f}")

    logging.info(f"Training accuracy: {accuracy:.2f}")
    logging.info("Training ended")

@cli.command()
def evaluate():
    """Evaluate trained model."""

    logging.info("Evaluation started")

    if not os.path.exists("model/trained_model.pkl"):
        click.echo("No trained model found. Run train first.")
        logging.warning("Evaluation failed because model does not exist.")
        return

    click.echo("Evaluating model...")

    X, y = load_iris(return_X_y=True)

    model = joblib.load("model/trained_model.pkl")

    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)

    click.echo(f"Overall accuracy: {accuracy:.2f}")

    logging.info(f"Evaluation accuracy: {accuracy:.2f}")
    logging.info("Evaluation ended")


if __name__ == "__main__":
    cli()