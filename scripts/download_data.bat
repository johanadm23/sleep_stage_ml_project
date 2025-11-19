@echo off
echo Creating data directories...
mkdir data\raw 2>nul

echo Downloading dataset...
curl -L https://physionet.org/static/published-projects/sleep-edfx/sleep-edf-database-expanded-1.0.0.zip

echo Unzipping...
powershell -Command "Expand-Archive -Path data\raw\dataset.zip -DestinationPath data\raw -Force"

echo Done!
