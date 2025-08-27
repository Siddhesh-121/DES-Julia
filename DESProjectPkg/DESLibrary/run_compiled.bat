@echo off
REM Simple batch script to run Julia with compiled system image
REM Usage: run_compiled.bat script.jl [args...]

set SYSIMAGE=DESLibrary_compiled.so

if exist "%SYSIMAGE%" (
    echo Running with compiled system image: %SYSIMAGE%
    julia --sysimage "%SYSIMAGE%" %*
) else (
    echo Compiled system image not found, running normally
    echo To compile: julia build_sysimage.jl
    julia %*
)