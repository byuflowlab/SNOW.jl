using GCopt
using Documenter

makedocs(;
    modules=[GCopt],
    authors="Andrew Ning <aning@byu.edu> and contributors",
    repo="https://github.com/byuflowlab/GCopt.jl/blob/{commit}{path}#L{line}",
    sitename="GCopt.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://byuflowlab.github.io/GCopt.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/byuflowlab/GCopt.jl",
)
