# Reinforcement Learning Control for Arm-X4

## Installation

1. Install IsaacSim [5.1.0](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html) by downloading release and unzip it to a desired location `$ISAACSIM_PATH`
2. Clone [Isaac Lab (2026/01/25)](https://github.com/isaac-sim/IsaacLab) and setup soft link forking to `$ISAACLAB_PATH` by
    ```bash
    cd IsaacLab
    ln -s $ISAACSIM_PATH _isaac_sim
    ```
3. Clone this repository to `${workspace}`
4. We leverage `uv` for python interpreter management. Install `uv` by following the [installation guide](https://docs.astral.sh/uv/getting-started/installation/) and setup `uv` environment:
    ```bash
    uv venv --python 3.11 lab
    source lab/bin/activate
    uv pip install -U pip
    ```
    Now the workspace should look like this:
    ```bash
    ${workspace}
    ├── lab/            # uv venv
    ├── Arm_X4/         # this repository
    ├── IsaacLab/       # Isaac Lab repository
    ├────_isaac_sim/    # soft link to IsaacSim
    ```
5. Activate `uv` environment and install IsaacLab and this repository:
    ```bash
    cd IsaacLab
    ./isaaclab.sh --uv ../lab       # link python interpreter to uv venv
    ./isaaclab.sh -i rsl_rl         # install IsaacLab and rsl_rl library

    cd Arm_X4
    python -m pip install -e source/Arm_X4   # install this repository in editable mode

## Task

- Listing the available tasks:
    ```bash
    python scripts/list_envs.py
    ```

- Running a task:

    ```bash
    python scripts/rsl_rl/train.py --task=Arm-X4
    ```

- Running a task with dummy agents:

    These include dummy agents that output zero or random agents. They are useful to ensure that the environments are configured correctly.

    - Zero-action agent

        ```bash
        python scripts/zero_agent.py --task=Arm-X4
        ```
    - Random-action agent

        ```bash
        python scripts/random_agent.py --task=Arm-X4
        ```

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/Arm_X4"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```
