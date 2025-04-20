import os
import subprocess
import sys
import tempfile
from pathlib import Path
from platform import machine

import numpy as np


# From https://github.com/LeoIV/BenchSuite/tree/master
class Mopta08:
    num_dim = 124

    def __init__(self):
        self.sysarch = 64 if sys.maxsize > 2**32 else 32
        self.machine = machine().lower()

        if self.machine == "armv7l":
            assert self.sysarch == 32, "Not supported"
            self._mopta_exectutable = "mopta08_armhf.bin"
        elif self.machine == "x86_64":
            assert self.sysarch == 64, "Not supported"
            self._mopta_exectutable = "mopta08_elf64.bin"
        elif self.machine == "i386":
            assert self.sysarch == 32, "Not supported"
            self._mopta_exectutable = "mopta08_elf32.bin"
        elif self.machine == "amd64":
            assert self.sysarch == 64, "Not supported"
            self._mopta_exectutable = "mopta08_amd64.exe"
        else:
            raise RuntimeError("Machine with this architecture is not supported")

        self._mopta_exectutable = os.path.join(Path(__file__).parent.parent, "data", "mopta08", self._mopta_exectutable)
        self.directory_file_descriptor = tempfile.TemporaryDirectory()
        self.directory_name = self.directory_file_descriptor.name

    def __call__(self, x):
        """
        Evaluate Mopta08 benchmark for one point
        :param x: one input configuration
        :return: value with soft constraints
        """

        assert np.all(x.min() > -1) and np.all(x.max() < 1)

        x = (1 + x) / 2

        x = x.flatten()
        assert len(x) == Mopta08.num_dim, len(x)
        # write input to file in dir
        with open(os.path.join(self.directory_name, "input.txt"), "w+") as tmp_file:
            for _x in x:
                tmp_file.write(f"{_x}\n")
        # pass directory as working directory to process
        popen = subprocess.Popen(
            self._mopta_exectutable,
            stdout=subprocess.PIPE,
            cwd=self.directory_name,
        )
        popen.wait()
        # read and parse output file
        with open(os.path.join(self.directory_name, "output.txt"), "r") as f:
            output = f.read().split("\n")
        output = [x.strip() for x in output]
        output = np.array([float(x) for x in output if len(x) > 0])
        value = output[0]
        constraints = output[1:]
        # see https://arxiv.org/pdf/2103.00349.pdf E.7
        return (value + 10 * np.sum(np.clip(constraints, a_min=0, a_max=None))).unsqueeze(-1)
