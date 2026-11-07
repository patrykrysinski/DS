# Pyenv setup for this project

This directory is configured to use pyenv for Python version management.

Recommended local Python version
- `.python-version` contains: `3.13.7` (you can change this to a different installed version)

Quick setup steps (macOS / zsh)

1. Install pyenv (if not already installed):

   - Using Homebrew:

     brew update
     brew install pyenv pyenv-virtualenv

2. Install the Python version listed in `.python-version` (if you don't have it):

   pyenv install 3.13.7

3. (Optional) Create a virtualenv for the project (using pyenv-virtualenv):

   pyenv virtualenv 3.13.7 react_tool_agent_project

4. Set local pyenv for this directory (two common options):

   - Use the raw version from `.python-version`:
     cd AI/react_tool_agent_project
   pyenv local 3.13.7

   - Or use created virtualenv:
     cd AI/react_tool_agent_project
     pyenv local react_tool_agent_project

5. Activate and install dependencies:

   # ensure the shims pick up the local version/venv
   exec "$SHELL"

   # install pinned dependencies
   pip install -r requirements.txt

Notes
- If `pyenv install` fails because of missing build dependencies, follow pyenv's macOS troubleshooting guide (install `openssl`, `readline`, `xz`, etc. via Homebrew).
- If you prefer `venv` instead of `pyenv-virtualenv`, create a venv with the configured Python after installing the requested version.

If you'd like, I can also:
- detect most-used packages in this project and pre-populate `requirements.txt`.
- create a proper virtualenv and install dependencies locally (requires running pyenv on your machine).
