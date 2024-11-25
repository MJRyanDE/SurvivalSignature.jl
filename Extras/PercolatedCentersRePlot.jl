using CairoMakie
using LaTeXStrings
using ColorSchemes

# Define error values for RMSE and RAE, just as an example, you'll need to insert your actual values
samples = [10, 50, 100, 500, 1000]

rmse_values = [
    [0.0705, 0.026, 0.0222, 0.011, 0.0101], # Grid
    [0.1054, 0.0258, 0.0181, 0.0121, 0.0099], # SparseGrid
    [0.1306, 0.0505, 0.0227, 0.0258, 0.013], # Greedy
    [0.0905, 0.0396, 0.0354, 0.0114, 0.0103], # GeometricGreedy
    [0.1318, 0.0534, 0.0212, 0.0138, 0.0119], # Leja
    [0.0567, 0.0231, 0.0159, 0.0116, 0.0111], # Grid_P
    [0.0681, 0.0279, 0.0197, 0.0266, 0.0102], # SparseGrid_P
    [0.1258, 0.0482, 0.0248, 0.0127, 0.0132], # Greedy_P
    [0.076, 0.0456, 0.0183, 0.0124, 0.0112], # GeometricGreedy_P
    [0.0856, 0.0272, 0.0195, 0.0126, 0.0128],  # Leja_P
]

rae_values = [
    [0.177, 0.0761, 0.0631, 0.0346, 0.0308], # Grid
    [0.2266, 0.0762, 0.0556, 0.0349, 0.03], # SparseGrid
    [0.2665, 0.1268, 0.0674, 0.0536, 0.0395], # Greedy
    [0.2223, 0.0961, 0.0804, 0.0349, 0.0312], # GeometricGreedy
    [0.2681, 0.1125, 0.0637, 0.0408, 0.0365], # Leja
    [0.157, 0.0756, 0.0512, 0.0366, 0.0348], # Grid_P
    [0.1787, 0.0796, 0.0601, 0.0551, 0.0308], # SparseGrid_P
    [0.2682, 0.1443, 0.0675, 0.0396, 0.042], # Greedy_P
    [0.1732, 0.1014, 0.0546, 0.0377, 0.0349], # GeometricGreedy_P
    [0.2044, 0.0853, 0.0587, 0.0383, 0.042],   # Leja_P
]

# Colors: Shades of blue for non-percolated, shades of red for percolated

blues = (colorschemes[:jblue])[1:5]
reds = (colorschemes[:jred])[1:5]

colors = vcat(blues..., reds...)

# Method names for hardcoding labels in LaTeX format
methods = [
    L"Grid",
    L"SparseGrid",
    L"Greedy",
    L"GeometricGreedy",
    L"Leja",
    L"Grid_P",
    L"SparseGrid_P",
    L"Greedy_P",
    L"GeometricGreedy_P",
    L"Leja_P",
]

# Create the subplot for RMSE
plt = Figure(; size=(600, 400), fonts=(; regular="CMU Serif"))
ax1 = Axis(
    plt[1, 1];
    title=LaTeXStrings.latexstring("RMSE"),
    xlabel=LaTeXStrings.latexstring("SAMPLES"),
    xgridstyle=:dash,
    ygridstyle=:dash,
    xtickalign=1,
    xticksize=5,
    ytickalign=1,
    yticksize=5,
    xscale=log10,
    xminorticksvisible=true,
    yscale=log10,
    yminorticksvisible=true,
)

for i in 1:length(rmse_values)
    lines!(ax1, samples, rmse_values[i]; label=methods[i], color=colors[i])
end

# Create the subplot for RAE
ax2 = Axis(
    plt[2, 1];
    title=LaTeXStrings.latexstring("RAE"),
    xlabel=LaTeXStrings.latexstring("SAMPLES"),
    xgridstyle=:dash,
    ygridstyle=:dash,
    xtickalign=1,
    xticksize=5,
    ytickalign=1,
    yticksize=5,
    xscale=log10,
    xminorticksvisible=true,
    yscale=log10,
    yminorticksvisible=true,
)
for i in 1:length(rae_values)
    lines!(ax2, samples, rae_values[i]; label=methods[i], color=colors[i])
end

Legend(plt[1:2, 2], ax1; unique=true, merge=true)

# Display the combined plot
display(plt)

save("percolated_centers_comparison_error.pdf", plt)