using Pkg
Pkg.add("PackageCompiler")

using PackageCompiler

create_sysimage([]; precompile_execution_file="test.jl", sysimage_path="test.so")