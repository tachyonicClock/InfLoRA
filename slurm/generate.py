from dataclasses import dataclass, asdict

from jinja2 import Environment, FileSystemLoader


@dataclass
class Job:
    dataset: str
    duration: str
    method: str
    memory: str = "8G"


def main():
    jobs = [
        Job("cifar100", "08:00:00" , "codap"),
        Job("cifar100", "08:00:00" , "dualprompt"),
        Job("cifar100", "08:00:00" , "inflora"),
        Job("cifar100", "08:00:00" , "l2p"),
        Job("domainnet", "12:00:00", "codap", memory="16G"),
        Job("domainnet", "12:00:00", "dualprompt", memory="16G"),
        Job("domainnet", "12:00:00", "inflora", memory="16G"),
        Job("domainnet", "12:00:00", "l2p", memory="16G"),
        Job("imagenetr", "13:00:00", "codap"),
        Job("imagenetr", "13:00:00", "dualprompt"),
        Job("imagenetr", "13:00:00", "inflora"),
        Job("imagenetr", "13:00:00", "l2p"),
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
