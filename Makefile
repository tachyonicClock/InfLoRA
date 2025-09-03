
.PHONY: fmt
fmt:
	ruff format
	ruff check --fix --extend-select I

link:
	ln -s ${ECS_SCRATCH}/log/inflora logs

launch:
	ts --set_gpu_free_perc=97
	ts -S 3
	ts -G 1 -m -L CIFAR100_inflora python main.py --device 0 --config configs/cifar100_inflora.yaml 
	ts -G 1 -m -L CIFAR100_inflora python main.py --device 0 --config configs/domainnet_inflora.yaml
	ts -G 1 -m -L CIFAR100_inflora python main.py --device 0 --config configs/mimg10_inflora.yaml 

clean-logs:
	rm -rf logs/*
