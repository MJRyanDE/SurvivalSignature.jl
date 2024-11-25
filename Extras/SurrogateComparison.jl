using CairoMakie
using LaTeXStrings

times2 = [0.11451345, 0.1234242, 0.22354123, 0.234124, 0.345455]
times1 = [0.1132, 0.2123, 0.3573234, 0.414252345, 0.545134]

function plotSurrogateComparison(times...)
    labels = [L"RBF", L"Kriging"]
    colors = [colorant"#557ec1", colorant"#ac2c71"]

    plt = Figure(; size=(600, 200), fonts=(; regular="CMU Serif"))
    ax = Axis(
        plt[1, 1];
        xgridstyle=:dash,
        xlabel=L"ITERATIONS",
        ylabel=L"Time\ (s)",
        ygridstyle=:dash,
        xtickalign=1,
        xticksize=5,
        ytickalign=1,
        yticksize=5,
        #xscale=log10,
        xminorticksvisible=true,
        yscale=log10,
        yminorticksvisible=true,
    )
    for (i, (time, color)) in enumerate(zip(times, colors))
        lines!(ax, time; label=labels[i], color=color, linewidth=1.5)
    end
    Legend(plt[1, 2], ax; unique=true, merge=true)
    return plt
end

plt = plotSurrogateComparison(times1, times2)

CairoMakie.display(plt)
