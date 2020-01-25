@echo off
sox --version
IF ERRORLEVEL 9009 set PATH=C:\Programs\sox-14.4.2;%PATH%
set DEEPSPEECH_MODEL=F:\temp\deepspeech-0.6.0-models
set DEEPSPEECH_LM=lm-cmd
node start.js