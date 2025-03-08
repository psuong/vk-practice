if (-Not (Test-Path -Path bin)) {
    New-Item .\bin -ItemType "directory"
}

if (-Not (Test-Path -Path bin/shaders)) {
    New-Item .\bin\shaders -ItemType "directory"
}

cmake --preset ninja
cmake --build --preset ninja-build

foreach ($shader in Get-ChildItem .\shaders -Filter *.comp) {
    $name = (Get-Item $shader).BaseName + ".spv"
    $target = ".\bin\shaders\$name"

    Write-Host "Compiling $shader"
    glslangValidator.exe -V $shader -o $target
}
