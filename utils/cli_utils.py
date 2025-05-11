import argparse

def parse_encode_args():
    """
    Parse command‐line args for train+embed script.
    --config       JSON config path
    --dataset      optional UCR dataset name to override data_folder
    --output‐prefix  prefix for saved .npy files
    """
    p = argparse.ArgumentParser(
        description="Train encoder & extract embeddings"
    )
    p.add_argument(
        "-c", "--config",
        default="configs/config_rnn_ae.json",
        help="path to JSON config"
    )
    p.add_argument(
        "-d", "--dataset",
        default="Adiac",
        help="override dataset name (sets data_folder=./data/<dataset>/numpy/)"
    )
    p.add_argument(
        "-o", "--output",
        default="data",
        help="prefix for saved embeddings filenames"
    )
    p.add_argument(
        "-e", "--epochs",
        type=int,
        default=10,
        help="number of training epochs"
    )
    return p.parse_args()