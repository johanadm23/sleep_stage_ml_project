@echo off
echo Downloading Sleep-EDF dataset...

set URL=https://physionet.org/static/published-projects/sleep-edfx/sleep-edf-database-expanded-1.0.0.zip
set OUT=..\data\raw\sleep_edf.zip

curl -L %URL% -o %OUT%

echo Unzipping...
powershell -command "Expand-Archive -Path %OUT% -DestinationPath ..\data\raw -Force"

echo Done!
pause

