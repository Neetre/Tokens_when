@ECHO OFF

CALL ./setup/setup_Windows.bat

CLS

REM Change directory to 'bin' and run the application
CD /d bin
IF ERRORLEVEL 1 (
    ECHO Failed to change directory to 'bin'
    PAUSE
    EXIT /B 1
)

python token_when.py --help

python token_when.py --load-mod

PAUSE

:End
EXIT /B 0