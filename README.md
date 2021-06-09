# Neural Pointers

## Setup

Either setup a local environment and install the necessary requirements via `pip install -r requirements.txt` (tested
with Python 3.8 so far), or use dedicated docker container setup, e.g.

```bash
docker run -it --name neural-pointers -v <path-to-local-workspace>:/mnt/workspace python:3.8 bash
cd /mnt/workspace
pip install -r requirements.txt
```

## Usage

An exemplary configuration to the simulation can be found in `./conf/`. The configuration is based on `hydra` and
hiearchically defined by the folder structure and files in the respective directory. To alter simulation parameters
any of the files can be adapted or duplicated and altered. Furthermore, `hydra` supports overriding of config
parameters either at runtime from within a python script, or while parsing the command line call to the main script.
Also, by using the command line functionality provided by `hydra` easy grid searches or sweeps over sets of parameters
(and their combinations) can be achieved as described in the
[docs](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/)). The output of each run can be found in a
respective subfolder in `./outputs/` containing the used `hydra` config for the simulation as well as the performed
visualizations.

To run a simulation simply call `python run_simulation.py`.

## Contributing

Make sure to keep the formatting nice and tidy by running `black .` in shell before committing to the project.
Docstring formatting is according to Google style, though not strictly enforced so far.
