from dataclasses import dataclass, asdict

from jinja2 import Environment, FileSystemLoader


@dataclass
class Job:
    dataset: str
    method: str
    memory: str = "4G"


def main():
    jobs = [
        Job("cifar100", "codap"),
        Job("cifar100", "dualprompt"),
        Job("cifar100", "inflora"),
        Job("cifar100", "l2p"),
        Job("domainnet", "codap", memory="16G"),
        Job("domainnet", "dualprompt", memory="16G"),
        Job("domainnet", "inflora", memory="16G"),
        Job("domainnet", "l2p", memory="16G"),
        Job("imagenetr", "codap"),
        Job("imagenetr", "dualprompt"),
        Job("imagenetr", "inflora"),
        Job("imagenetr", "l2p"),
    ]
    env = Environment(loader=FileSystemLoader("slurm"))
    template = env.get_template("template.sl.jinja")
    for job in jobs:
        script_content = template.render(asdict(job))
        script_filename = f"sbatch/{job.dataset}_{job.method}.sl"
        with open(script_filename, "w") as f:
            f.write(script_content)
        print(f"+ {script_filename}")


if __name__ == "__main__":
    main()
