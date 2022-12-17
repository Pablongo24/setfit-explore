# exit when any command fails
set -e

current_path="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

#echo 'Check that Docker Container builds...'
#docker compose -f docker-compose.local.yml build ci-cd

echo 'Run Black Formatter...'
python -m black src/ --config=pyproject.toml

echo 'Run Flake8 Linter - Find syntax errors or undefined names...'
python -m flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics
echo 'Run Flake8 Linter - Find lines longer than 127 chars. Exits zero, just a counter of long lines...'
python -m flake8 src/ --count --exit-zero --max-line-length=127 --statistics
echo 'Run Flake8 Linter - Enforce good linting, ignores long lines (--max-line-length)...'
python -m flake8 src/ --max-complexity 10 --ignore E501

echo 'Run Tests...'
rm -f "$current_path"/coverage.xml
touch "$current_path"/coverage.xml
python -m pytest --cov=src/ --cov-config=.coveragerc --cov-report=xml --cov-report=term --cov-fail-under=95