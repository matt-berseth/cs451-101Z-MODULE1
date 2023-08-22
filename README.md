# cs451-101Z-MODULE1
This repo contains some examples we will walk through in class for module 1.

## Windows 10/11 Steps
```powershell
# clone this repo
git clone https://github.com/matt-berseth/cs451-101Z-MODULE1.git

# path into the repo
cd cs452-101Z-MODULE1

# create the virtual env
python3.10 -m venv .venv

# activate
.\.venv\Scripts\activate.ps1

# install the deps
python -m pip install --upgrade pip
pip install -r requirements.txt

# launch vscode
code .
```

## Ubuntu 22.04 Steps
```bash
# clone this repo
git clone https://github.com/matt-berseth/cs451-101Z-MODULE1.git

# path into the repo
cd cs452-101Z-MODULE1

# create the virtual env
python3 -m venv .venv
source ./.venv/bin/activate

# install the deps
pip install --upgrade pip
pip install -r requirements.txt

# launch vscode
code .
```