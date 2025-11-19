@echo off
echo Downloading Sleep-EDF dataset...

set URL=https://physionet.org/content/sleep-edf/get-zip/1.0.0/
set OUT=..\data\raw\sleep_edf.zip

curl -L %URL% -o %OUT%

echo Unzipping...
powershell -command "Expand-Archive -Path %OUT% -DestinationPath ..\data\raw -Force"

echo Done!
pause

