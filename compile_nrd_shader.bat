@echo off
echo Compiling NRD Temporal Accumulation Shader...

REM 检查glslc是否存在
where glslc >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: glslc not found in PATH!
    echo Please install Vulkan SDK or add glslc to your PATH.
    pause
    exit /b 1
)

REM 编译compute shader
glslc -fshader-stage=compute shaders\nrd_temporal_accumulation.comp -o shaders\nrd_temporal_accumulation.comp.spv

if %ERRORLEVEL% EQU 0 (
    echo Shader compiled successfully!
    echo Output: shaders\nrd_temporal_accumulation.comp.spv
) else (
    echo Shader compilation failed!
    pause
    exit /b 1
)

pause
