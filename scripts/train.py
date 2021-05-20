import os
import click
import logging
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def load_dataset(train_path, test_path, tokenizer):
    train_dataset = TextDataset(
        tokenizer=tokenizer, file_path=train_path, block_size=128
    )

    test_dataset = TextDataset(tokenizer=tokenizer, file_path=test_path, block_size=128)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    return train_dataset, test_dataset, data_collator


@click.command()
@click.option("--train_dir", help="Path to save training files")
@click.option("--output_dir", help="Path to save checkpoints")
@click.option("--log_level", default="INFO", help="Log level (default: INFO)")
def main(train_dir, output_dir, log_level):

    # Set logger config
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Load a tokenizer
    logging.info("Loading the tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Load GPT2 model
    logging.info("Loading the model...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Load the dataset
    train_path = os.path.join(train_dir, "train.txt")
    test_path = os.path.join(train_dir, "test.txt")
    logging.info("Loading the dataset...")
    train_dataset, test_dataset, data_collator = load_dataset(
        train_path, test_path, tokenizer
    )

    # Define training arguments
    logging.info("Defining training args...")
    training_args = TrainingArguments(
        output_dir=output_dir,  # The output directory
        overwrite_output_dir=True,  # overwrite the content of the output directory
        num_train_epochs=3,  # number of training epochs
        per_device_train_batch_size=16,  # batch size for training
        per_device_eval_batch_size=32,  # batch size for evaluation
        eval_steps=400,  # Number of update steps between two evaluations.
        save_steps=800,  # after # steps model is saved
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        prediction_loss_only=True,
    )

    # Initialise trainer
    logging.info("Initialising trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    if os.path.exists(output_dir):
        ordered_checkpoints = sorted(
            os.listdir(output_dir), key=lambda x: int(x.split("-")[-1])
        )
        latest_checkpoint = os.path.join(output_dir, ordered_checkpoints[-1])

        # Train the model
        logging.info(f"Restarting the model from checkpoint {latest_checkpoint}")
        trainer.train(latest_checkpoint)

    else:
        # Train the model
        logging.info("Training the model...")
        trainer.train()

    # Save the model
    logging.info("Saving the model...")
    trainer.save_model()


if __name__ == "__main__":
    main()
