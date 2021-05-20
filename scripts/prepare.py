import os
import click
import logging
import numpy as np
import pandas as pd


# Set random seed
np.random.seed(1234)


@click.command()
@click.option("--source_csv", help="Path to source csv")
@click.option("--train_dir", help="Path to save training files")
@click.option(
    "--test_size", default=0.15, type=float, help="Path to save training files"
)
@click.option("--log_level", default="INFO", help="Log level (default: INFO)")
def main(source_csv, train_dir, test_size, log_level):

    # Set logger config
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Load the data
    herald_data = pd.read_csv(source_csv, low_memory=False)

    # Drop missing articles
    herald_data = herald_data[~herald_data.paragraphs.isna()]

    # Perform train-test split
    herald_data["seed"] = np.random.rand(len(herald_data))
    herald_data["train_set"] = herald_data["seed"].apply(
        lambda x: "train" if x < 1 - test_size else "test"
    )

    train_data_path = "data/train"
    for train_set in ["train", "test"]:
        output_path = os.path.join(train_data_path, train_set + ".txt")
        logging.info("Saving {} data to {}".format(train_set, output_path))
        with open(output_path, "w") as f:
            for line in herald_data.loc[
                herald_data["train_set"] == train_set, "paragraphs"
            ]:
                f.write(line + "\n")

    logging.info(
        "Train dataset length: {}".format(np.sum(herald_data.train_set == "train"))
    )
    logging.info(
        "Test dataset length: {}".format(np.sum(herald_data.train_set == "test"))
    )


if __name__ == "__main__":
    main()
