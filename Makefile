
.PHONY: fmt
fmt:
	ruff format
	ruff check --fix --extend-select I

link:
	ln -s ${ECS_SCRATCH}/log/inflora logs

clean-logs:
	rm -rf logs/*
