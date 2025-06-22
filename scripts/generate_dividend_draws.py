import csv
import pathlib
from typing import Union


def main(path: Union[str, pathlib.Path] = "data/dividend_draws.csv") -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["period", "dividend"])
        writer.writerows([[1, 1.0], [2, 1.1], [3, 0.9]])


if __name__ == "__main__":
    main()
