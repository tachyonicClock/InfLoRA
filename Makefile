
.PHONY: fmt
fmt:
	ruff format
	ruff check --fix --extend-select I

link:
	ln -s ${ECS_SCRATCH}/log/inflora logs

launch:
	ts --set_gpu_free_perc=97
	ts -S 3
# 	CIFAR100 Launch:
	ts -G 1 -m -L cifar100_inflora python main.py --device 0 --config configs/cifar100_inflora.yaml
	ts -G 1 -m -L cifar100_codap   python main.py --device 0 --config configs/cifar100_codap.yaml

# 	ImageNet-R Launch:
	ts -G 1 -m -L mimg10_inflora    python main.py --device 0 --config configs/mimg10_inflora.yaml 
	ts -G 1 -m -L mimg10_codap      python main.py --device 0 --config configs/mimg10_codap.yaml

# 	DomainNet Launch:
	ts -G 1 -m -L domainnet_inflora python main.py --device 0 --config configs/domainnet_inflora.yaml
	ts -G 1 -m -L domainnet_codap   python main.py --device 0 --config configs/domainnet_codap.yaml

clean-logs:
	rm -rf logs/*
