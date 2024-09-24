using Pkg
Pkg.add("PackageCompiler")

using PackageCompiler

# nemoj ovo pozivat ako ti je Boga milo

create_sysimage(precompile_execution_file="ManualMirrorPlacement/test.jl", sysimage_path="ManualMirrorPlacement/test.so", include_transitive_dependencies=false)