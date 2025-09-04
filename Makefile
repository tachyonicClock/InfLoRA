
.PHONY: fmt
fmt:
	ruff format
	ruff check --fix --extend-select I

link:
	ln -s ${ECS_SCRATCH}/log/inflora logs

generate-slurm:
	python slurm/generate.py

launch:
	ts --set_gpu_free_perc=97
	ts -S 3
# 	CIFAR100 Launch:
	ts -G 1 -m -L cifar100_inflora    python main.py --device 0 --config configs/cifar100_inflora.yaml
	ts -G 1 -m -L cifar100_codap      python main.py --device 0 --config configs/cifar100_codap.yaml
	ts -G 1 -m -L cifar100_dualprompt python main.py --device 0 --config configs/cifar100_dualprompt.yaml
	ts -G 1 -m -L cifar100_l2p        python main.py --device 0 --config configs/cifar100_l2p.yaml

# 	ImageNet-R Launch:
	ts -G 1 -m -L imagenetr_inflora    python main.py --device 0 --config configs/imagenetr_inflora.yaml 
	ts -G 1 -m -L imagenetr_codap      python main.py --device 0 --config configs/imagenetr_codap.yaml
	ts -G 1 -m -L imagenetr_dualprompt python main.py --device 0 --config configs/imagenetr_dualprompt.yaml
	ts -G 1 -m -L imagenetr_l2p        python main.py --device 0 --config configs/imagenetr_l2p.yaml

# 	DomainNet Launch:
	ts -G 1 -m -L domainnet_inflora    python main.py --device 0 --config configs/domainnet_inflora.yaml
	ts -G 1 -m -L domainnet_codap      python main.py --device 0 --config configs/domainnet_codap.yaml
	ts -G 1 -m -L domainnet_dualprompt python main.py --device 0 --config configs/domainnet_dualprompt.yaml
	ts -G 1 -m -L domainnet_l2p        python main.py --device 0 --config configs/domainnet_l2p.yaml
	

clean-logs:
	rm -rf logs/*
