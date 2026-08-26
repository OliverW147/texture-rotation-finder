@echo off
setlocal

cd /d "%~dp0"

REM CPU version (always built)
echo [CPU] Compiling tex_match.exe...
gcc -O3 -o tex_match.exe tex_match.c -lm
if errorlevel 1 ( echo CPU build failed & exit /b 1 )
echo Done: tex_match.exe

REM GPU version (requires CUDA + MSVC)
where nvcc >nul 2>&1
if errorlevel 1 (
    echo nvcc not found -- skipping GPU build
    goto :eof
)

REM Locate the MSVC environment nvcc needs as its host compiler. CUDA supports
REM only VS 2019-2022, so skip any newer install that happens to be present.
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if "%VSCMD_VER%"=="" if exist "%VSWHERE%" (
    for /f "usebackq tokens=*" %%i in (`^""%VSWHERE%" -latest -version "[16.0,18.0)" -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath^"`) do (
        call "%%i\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
    )
)

REM Build a fatbin covering every consumer arch back to Turing, plus PTX for
REM the newest one so future cards can JIT. PTX only forward-JITs inside its
REM own architecture family, so each family needs its own entry.
REM Override with a single arch to build faster, e.g.
REM   set CUDA_GENCODE=-arch=sm_89 ^&^& build_tex.bat
if "%CUDA_GENCODE%"=="" set CUDA_GENCODE=-gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120 -gencode arch=compute_120,code=compute_120

echo [GPU] Compiling tex_match_gpu.exe...
nvcc -O2 %CUDA_GENCODE% --fmad=false ^
  -Xcompiler "/wd4244 /wd4267 /wd4101 /wd4996 /D_CRT_SECURE_NO_WARNINGS" ^
  -o tex_match_gpu.exe tex_match_gpu.cu
if errorlevel 1 ( echo GPU build failed & exit /b 1 )
echo Done: tex_match_gpu.exe
