param (
    [Parameter(Mandatory=$true)][string]$type = "slang"
)

if (-Not (Test-Path -Path bin)) {
    New-Item .\bin -ItemType "directory"
}

if (-Not (Test-Path -Path bin/shaders)) {
    New-Item .\bin\shaders -ItemType "directory"
}

cmake --preset ninja
cmake --build --preset ninja-build

if ($type -eq "glsl") {
    Write-Host "Compiling glsl shaders"
    foreach ($shader in Get-ChildItem .\shaders -Filter *.comp) {
        $name = (Get-Item $shader).BaseName + ".spv"
        $target = ".\bin\shaders\$name"

        Write-Host "Compiling $shader"
        glslangValidator.exe -V $shader -o $target
    }
} else {
    Write-Host "Compiling slang shaders"
    foreach ($shader in Get-ChildItem .\shaders -Filter *.slang) {
        $name = (Get-Item $shader).BaseName;

        if ($shader.Name.Contains(".comp")) {
            $target = ".\bin\shaders\$name.spv"
            Write-Host "Compiling Compute Shader: $shader to $target"
            slangc $shader -target spirv -profile cs_6_0 -o $target
        } else {
            $vert = ".\bin\shaders\$name" + "_vert.spv"
            Write-Host "Compiling Vertex Shader: $shader to $vert"
            slangc $shader -target spirv -profile vs_6_0 -entry vert -o $vert

            $frag = ".\bin\shaders\$name" + "_frag.spv"
            Write-Host "Compiling Fragment Shader: $shader to $frag"
            slangc $shader -target spirv -profile ps_6_0 -entry frag -o $frag
        }
    }
}
