Welcome to MuJoCo version 2.0.

The full documentation is available at http://www.mujoco.org/book
Here we provide brief notes to get you started:

The activation key (which you should have received with your license) is a
plain-text file whose path must be passed to the mj_activate() function.
The code samples assume that it is called mjkey.txt in the bin directory.

Once you have mjkey.txt in the bin directory, run:
  simulate ../model/humanoid.xml  (or ./simulate on Linux and macOS)
to see MuJoCo in action.

In general, the directory structure we have provided is merely a suggestion;
feel free to re-organize it if needed. MuJoCo does not have an installer
and does not write any files outside the executable directory.

The makefile in the sample directory generates binaries in the bin directory.
These binaries are pre-compiled and included in the software distribution.

Additional models are available at http://www.mujoco.org/forum under Resources.
