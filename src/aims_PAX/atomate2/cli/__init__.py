"""Helper functions for CLI."""
import argparse


def get_args_parser(description: str = "Aims-PAX CLI"):
    """Returns an argument parser."""
    parser = argparse.ArgumentParser(
        description=description
    )
    parser.add_argument(
        "--model-settings",
        type=str,
        default="./model.yaml",
        help="Path to model settings file",
    )
    parser.add_argument(
        "--aimsPAX-settings",
        type=str,
        default="./aimsPAX.yaml",
        help="Path to aimsPAX settings file",
    )
    return parser