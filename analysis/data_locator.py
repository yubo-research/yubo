import os


class DataLocator:
    def __init__(self, results_path):
        self.results = results_path

    def problems_in(self, exp_tag):
        return sorted(os.listdir(f"{self.results}/{exp_tag}"))

    def optimizers_in(self, exp_tag, problem):
        return sorted(os.listdir(f"{self.results}/{exp_tag}/{problem}"))

    def __call__(self, exp_tag, problem_name, optimizer_name):
        return f"{self.results}/{exp_tag}/{problem_name}/{optimizer_name}"
