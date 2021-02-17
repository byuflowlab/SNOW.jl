using SNOW
using Documenter

makedocs(;
    modules=[SNOW],
    authors="Andrew Ning <aning@byu.edu> and contributors",
    repo="https://github.com/byuflowlab/SNOW.jl/blob/{commit}{path}#L{line}",
    sitename="SNOW.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://byuflowlab.github.io/SNOW.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/byuflowlab/SNOW.jl",
)
