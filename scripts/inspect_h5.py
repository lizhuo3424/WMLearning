"""Print an HDF5 trajectory tree with shapes and dtypes."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory", type=Path)
    args = parser.parse_args()
    if not args.trajectory.is_file():
        raise FileNotFoundError(args.trajectory)

    with h5py.File(args.trajectory, "r") as handle:
        print(f"file={args.trajectory}")
        print(f"root_attributes={dict(handle.attrs)}")

        def show(name: str, obj: h5py.Group | h5py.Dataset) -> None:
            if isinstance(obj, h5py.Dataset):
                print(f"{name}: shape={obj.shape}, dtype={obj.dtype}")
            else:
                print(f"{name}/")

        handle.visititems(show)


if __name__ == "__main__":
    main()

