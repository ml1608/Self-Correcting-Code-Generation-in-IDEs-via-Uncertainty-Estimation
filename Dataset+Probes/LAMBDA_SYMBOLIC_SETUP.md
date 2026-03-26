## Lambda Setup: Symbolic-Execution Probe Training (Strict)

This setup is required for `train_probes.py` now that clustering is symbolic-only.

### 1) Create/activate environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2) Install dependencies

```bash
pip install -r "Dataset+Probes/requirements.txt"
```

### 3) Provide symbolic clustering helper

`train_probes.py` requires:

- env var: `PYEXZ3_CLUSTER_SCRIPT`
- value: absolute path to your symbolic clustering helper script

Example:

```bash
export PYEXZ3_CLUSTER_SCRIPT="/home/ubuntu/Self-Correcting-Code-Generation-in-IDEs-via-Uncertainty-Estimation/Dataset+Probes/pyexz3_cluster_helper.py"
```

### 4) Quick preflight checks

```bash
python "$PYEXZ3_CLUSTER_SCRIPT" --self-check
```

### 5) Run training

```bash
python "Dataset+Probes/train_probes.py"
```

If anything is missing, the script now fails fast with explicit setup errors.
