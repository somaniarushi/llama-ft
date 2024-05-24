import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def split_parquet_into_train_and_test(
    input_parquet: str,
    output_train_parquet: str,
    output_test_parquet: str,
) -> None:
    """
    Given a parquet file, splits the data into train and test sets.
    """
    assert Path(input_parquet).exists(), f"File {input_parquet} does not exist"
    df = pd.read_parquet(input_parquet)
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    train_df.to_parquet(output_train_parquet)
    test_df.to_parquet(output_test_parquet)

if __name__ == "__main__":
    split_parquet_into_train_and_test(
        input_parquet="data/intel_orca/0000.parquet",
        output_train_parquet="data/intel_orca/train.parquet",
        output_test_parquet="data/intel_orca/test.parquet",
    )