import os
import glob
from datasets import load_dataset, DatasetDict


def restore_dataset():
    csv_files = glob.glob("data/combined/*.csv")

    if not csv_files:
        print("Error: No CSV file found in the data/combined folder.")
        return

    csv_path = csv_files[0]
    print(f"CSV file found: {csv_path}")
    print("Restoring dataset...")

    dataset = load_dataset("csv", data_files=csv_path)

    train_test = dataset['train'].train_test_split(test_size=0.2, seed=42)
    test_val = train_test['test'].train_test_split(test_size=0.5, seed=42)

    final_dataset = DatasetDict({
        'train': train_test['train'],
        'validation': test_val['train'],
        'test': test_val['test']
    })

    output_path = "data/combined/combined_dataset"
    final_dataset.save_to_disk(output_path)

    print(f"\nSuccessfully restored dataset: {output_path}")
    print("You can now start training.")


if __name__ == "__main__":
    restore_dataset()