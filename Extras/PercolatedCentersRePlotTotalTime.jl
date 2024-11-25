using CairoMakie
using LaTeXStrings
using ColorSchemes

# Define total time values, replace with actual values as needed
samples = [10, 50, 100, 500, 1000]

adaptive_refinement_times = [
    [3.792467, 1.694753, 2.398559, 4.405154, 7.2774], # Grid
    [18.345774, 3.099485, 5.636987, 9.222651, 8.699372], # SparseGrid
    [153.962956, 112.481408, 29.680629, 46.824122, 58.570059], # Greedy
    [0.523519, 26.828274, 18.50584, 16.797016, 26.855489], # GeometricGreedy
    [202.042008, 47.558176, 13.713039, 41.017422, 40.016575], # Leja
    [1.909557, 1.259108, 1.971715, 1.147056, 2.417654], # Grid_P
    [2.585731, 1.470715, 3.028459, 6.65314, 8.035415], # SparseGrid_P
    [123.055695, 1.599726, 50.701609, 8.751571, 12.674628], # Greedy_P
    [4.609283, 23.289346, 3.49643, 2.689383, 6.307253], # GeometricGreedy_P
    [7.539077, 3.273687, 8.789625, 9.143795, 11.031321],  # Leja_P
]

total_time_values = [
    [4.179407, 1.826098, 2.638569, 5.178847, 8.180809], # Grid
    [19.243228, 3.238517, 5.862147, 9.73793, 9.612357], # SparseGrid
    [164.796816, 123.09141, 39.539925, 58.941845, 72.915112], # Greedy
    [1.250026, 27.831215, 19.424119, 17.967399, 28.433567], # GeometricGreedy
    [203.544793, 48.869802, 14.605328, 42.29055, 41.608312], # Leja
    [2.039117, 1.367603, 2.125054, 1.599718, 3.248635], # Grid_P
    [2.754533, 1.577956, 3.196545, 7.11841, 8.887623], # SparseGrid_P
    [134.089133, 12.080934, 61.644144, 20.760072, 26.907066], # Greedy_P
    [5.401846, 24.238038, 4.243442, 3.753309, 7.833633], # GeometricGreedy_P
    [8.2058, 3.966714, 9.529736, 10.197402, 12.491675],  # Leja_P
]

# Colors: Shades of blue for non-percolated, shades of red for percolated
blues = (colorschemes[:jblue])[1:5]
reds = (colorschemes[:jred])[1:5]

colors = vcat(blues..., reds...)

# Method names for hardcoding labels in LaTeX format
methods = [
    L"Grid",
    L"SGrid",
    L"G",
    L"GeoG",
    L"Leja",
    L"Grid _P",
    L"SGrid _P",
    L"G _P",
    L"GeoG _P",
    L"Leja _P",
]
# Create the subplot for Total Time
plt = Figure(; size=(600, 200), fonts=(; regular="CMU Serif"))
ax = Axis(
    plt[1, 1];
    xlabel=LaTeXStrings.latexstring("SAMPLES"),
    ylabel=LaTeXStrings.latexstring("Time\\ (s)"),
    xgridstyle=:dash,
    ygridstyle=:dash,
    xtickalign=1,
    xticksize=5,
    ytickalign=1,
    yticksize=5,
    xscale=log10,
    xminorticksvisible=true,
    yminorticksvisible=true,
)

for i in 1:length(total_time_values)
    lines!(ax, samples, total_time_values[i]; label=methods[i], color=colors[i])
end

Legend(plt[1, 2], ax; unique=true, merge=true, nbanks=2)

# Display the plot
display(plt)
save("percolated_centers_comparison_time.pdf", plt)