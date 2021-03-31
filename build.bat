@echo off

call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" x86_amd64


:: swap /GL for /Zi in debug builds, actually remove most of these for debug builds

set opts=/FC /GR- /EHa- /nologo /std:c17 /O2 /Oi /GL /favor:AMD64 /Qpar 
set opts=%opts% /W4 /wd4310 /wd4100 /wd4201 /wd4505 /wd4996 /wd4127 /wd4510 /wd4512 /wd4610 /wd4457 /WX /permissive-
set opts=%opts% /openmp

:: set opts=/FC /GR- /EHa- /nologo /std:c17 /Z7 /DDEBUG
:: debug

set code=%cd%
pushd bin
cl %opts% %code%\GA.c -FeGA.exe
popd