# airbnb_prices
[![Linter Actions Status](https://github.com/Seon82/projet_appr_auto/actions/workflows/lint.yml//badge.svg?branch=master)](https://github.com/Seon82/projet_appr_auto/actions)

airbnb_prices is a python package offering various utilities to help predict the price of Airbnb rentals. 

## Quickstart

```python
from airbnb_prices import DataPipeline
from airbnb_prices.data import download_dataset

# Download the dataset
dowload_dataset("./data/train_airbnb_berlin.csv")

# Preprocess the data using parameters from the config file
pipeline = DataPipeline.from_file("./data/train_airbnb_berlin.csv", "./examples/config.json")
pipeline.run()

X_train, y_train = pipeline.train_data
```
See the [examples](./examples) directory for more detailed examples, including a full training pipeline.

The package also exposes a CLI for quick model training, run `python airbnb_prices --help` for more information.

## Install
* Clone the repository.
* Run `pip install .` from the project's root.

## Contributing
* We use [poetry](https://github.com/python-poetry/poetry) for dependency management. The project can be installed in development mode by running `poetry install`.
* Code must be formatted using `black` and `isort`.
* Use `pylint` to check code quality before committing.

## Links

Find the report for this project on overleaf.

Link to edit the report: https://www.overleaf.com/3389558812mrxxwvwnvfpd Link to view the report only: https://www.overleaf.com/read/vgmnnsqnfjdt
