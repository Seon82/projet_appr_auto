default:
  just --list

lint:
  poetry run pylint airbnb_prices

fmt:
  poetry run isort airbnb_prices
  poetry run black airbnb_prices 