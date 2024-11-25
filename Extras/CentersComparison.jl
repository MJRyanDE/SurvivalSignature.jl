using CairoMakie
using LaTeXStrings

rsme_error = [
    0.1469,
    0.0962,
    0.0589,
    0.0301,
    0.0223,
    0.0099,
    0.0098,
    0.0078,
    0.008,
    0.0073,
    0.0067,
    0.0075,
    0.0089,
    0.0076,
    0.0068,
    0.006,
    0.006,
    0.0068,
    0.0069,
    0.0074,
    0.0068,
    0.008,
    0.0069,
    0.0086,
    0.0067,
    0.0059,
    0.0073,
    0.0069,
    0.0085,
    0.0061,
    0.0063,
    0.0065,
    0.0085,
    0.006,
    0.008,
    0.0063,
    0.0059,
    0.0068,
    0.0064,
]

rae_error = [
    0.7548,
    0.2869,
    0.1876,
    0.0944,
    0.0698,
    0.0301,
    0.0309,
    0.0234,
    0.0243,
    0.0234,
    0.0217,
    0.0275,
    0.0271,
    0.0234,
    0.0216,
    0.0188,
    0.0187,
    0.0209,
    0.0216,
    0.0229,
    0.0219,
    0.025,
    0.0215,
    0.0291,
    0.0207,
    0.0186,
    0.0217,
    0.0222,
    0.0262,
    0.0195,
    0.0208,
    0.0193,
    0.0283,
    0.0195,
    0.0248,
    0.0195,
    0.0186,
    0.0207,
    0.0201,
]

total_time = [
    5.013719,
    2.612391,
    4.504187,
    4.368656,
    2.036617,
    2.554282,
    3.517313,
    5.403278,
    5.153162,
    5.89506,
    6.088986,
    6.114322,
    3.454606,
    3.944637,
    6.661725,
    6.811212,
    6.777253,
    6.939796,
    7.384399,
    8.057058,
    7.697099,
    8.05594,
    8.263003,
    9.000696,
    7.288758,
    8.70701,
    9.186507,
    11.268166,
    13.034276,
    12.404216,
    17.512353,
    19.141578,
    17.135938,
    18.290152,
    20.089139,
    19.564636,
    21.504312,
    24.28534,
    26.192477,
]

centers = [
    4,
    9,
    16,
    25,
    36,
    49,
    64,
    81,
    100,
    121,
    144,
    169,
    196,
    225,
    256,
    289,
    324,
    361,
    400,
    441,
    484,
    529,
    576,
    625,
    676,
    729,
    784,
    841,
    900,
    961,
    #1024,
    #1089,
    #1156,
    #1225,
]

vector1 = [
    0.1478,
    0.0976,
    0.0571,
    0.0314,
    0.0155,
    0.0103,
    0.0085,
    0.0095,
    0.0083,
    0.0084,
    0.008,
    0.0068,
    0.0073,
    0.0071,
    0.0075,
    0.0074,
    0.007,
    0.0067,
    0.0065,
    0.0066,
    0.0065,
    0.007,
    0.0067,
    0.0064,
    0.0064,
    0.0072,
    0.0075,
    0.0068,
    0.0064,
    0.0063,
    #0.0067,
    #0.0062,
    #0.0065,
    #0.007,
]

vector2 = [
    0.7533,
    0.2752,
    0.1818,
    0.1009,
    0.0506,
    0.035,
    0.0274,
    0.0284,
    0.0261,
    0.0258,
    0.0264,
    0.0214,
    0.0233,
    0.0218,
    0.0251,
    0.0247,
    0.0218,
    0.0208,
    0.0205,
    0.0204,
    0.0199,
    0.0216,
    0.0209,
    0.0199,
    0.0199,
    0.0224,
    0.0228,
    0.0208,
    0.0201,
    0.02,
    0.0214,
    0.0192,
    0.0204,
    0.0225,
]

vector3 = [
    5.9819,
    2.8681,
    5.7775,
    3.4866,
    4.8808,
    4.4095,
    5.4179,
    6.0488,
    6.0958,
    4.8838,
    7.4444,
    13.3287,
    9.6508,
    10.4021,
    11.0987,
    21.2486,
    20.1794,
    17.3808,
    26.2719,
    28.5214,
    26.8904,
    19.8329,
    29.8363,
    40.995,
    32.5045,
    27.7507,
    38.7931,
    70.0274,
    86.9223,
    127.6082,
    #125.7189,
    #215.3956,
    #203.8637,
    #175.8191,
]

vector4 = [
    0.1459,
    0.0974,
    0.0579,
    0.0311,
    0.0164,
    0.0106,
    0.0097,
    0.0113,
    0.0083,
    0.0082,
    0.0087,
    0.0083,
    0.0083,
    0.0113,
    0.0087,
    0.0081,
    0.0079,
    0.0081,
    0.0083,
    0.0081,
    0.0079,
    0.0081,
    0.0079,
    0.0082,
]

# Second vector
vector5 = [
    0.6902,
    0.2852,
    0.1853,
    0.0978,
    0.0513,
    0.0318,
    0.031,
    0.0364,
    0.0249,
    0.0246,
    0.0265,
    0.0248,
    0.0256,
    0.0381,
    0.0286,
    0.0245,
    0.0237,
    0.0247,
    0.0251,
    0.0246,
    0.0238,
    0.0245,
    0.0238,
    0.0244,
]

# Third vector
vector6 = [
    5.2448,
    3.1788,
    4.508,
    4.1631,
    4.3713,
    3.4143,
    3.7375,
    4.5835,
    5.5312,
    5.8269,
    5.3513,
    6.1162,
    5.853,
    5.8763,
    6.1917,
    6.6873,
    6.5667,
    7.2076,
    7.7898,
    8.0975,
    7.5498,
    8.4295,
    9.1437,
    9.8666,
]

candidates = 2500

centers = centers ./ candidates * 100

# Plotting the results
# Double-Axis Plot

colors = [colorant"#557ec1", colorant"#ac2c71"]

f = Figure(; size=(600, 300), fonts=(; regular="CMU Serif"))

ax1 = Axis(
    f[1, 1];
    xlabel="Percent of State Vectors (%)",
    ylabel="Time (s)",
    ylabelcolor=colorant"#557ec1",
    ytickcolor=colorant"#557ec1",
    yticklabelcolor=colorant"#557ec1",
    leftspinecolor=colorant"#557ec1",
    rightspinecolor=colorant"#ac2c71",
    xticks=LinearTicks(10),
    yticks=LinearTicks(5),
)
ax2 = Axis(
    f[1, 1];
    ylabel="Root Mean Squared Error (RMSE)",
    yscale=log10,
    yticklabelcolor=colorant"#ac2c71",
    ytickcolor=colorant"#ac2c71",
    ylabelcolor=colorant"#ac2c71",
    flip_ylabel=true,
    yaxisposition=:right,
)
hidespines!(ax2)
hidexdecorations!(ax2)

lines!(ax1, centers, vector3; color=colorant"#557ec1")
lines!(ax2, centers, vector1; color=colorant"#ac2c71", label="RMSE")
# # Add RAE error to the main plot
# plot!(twinx(), rae_error; xaxis=:log10, label="RAE Error")
# plot!(twinx(), rae_error; xaxis=:log10, label="RAE Error")

# Add total time with a secondary y-axis

# Display the plot
display(f)

CairoMakie.save("centers_comparison.pdf", f)