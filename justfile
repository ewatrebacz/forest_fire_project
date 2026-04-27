[doc('List all recipes')]
default:
    just --list

check:
    @echo "Checking code..."
    poetry run ruff check . || ERROR=$?
    poetry run ruff format --check . || ERROR=$?
    poetry run mypy --incremental --show-error-codes --pretty --check-untyped-defs . || ERROR=$?


format:
    @echo "Formatting code..."
    poetry run ruff format . || ERROR=$?
    poetry run ruff check --fix || ERROR=$?

# This create sync notebooks and scripts, and create new notebooks from scripts
# It does not create new scripts from notebooks
default_report_dir := "report"
sync-notebooks:
    @echo "Syncing notebooks with scripts..."
    poetry run jupytext --sync ./script/*

sync-scripts:
    @echo "Syncing scripts with notebooks..."
    poetry run jupytext --sync ./notebook/*.ipynb


alias c := check
alias f := format
alias sn := sync-notebooks
alias ss := sync-scripts
