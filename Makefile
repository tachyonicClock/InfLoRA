
.PHONY: fmt
fmt:
	ruff format
	ruff check --fix --extend-select I
