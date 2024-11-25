module SurvivalSignatureUtils

using Statistics

__precompile__()

function ensure_column_array!(arr::Array)
    # If it's a 1D array, reshape it into a column vector
    if ndims(arr) == 1
        return reshape(arr, :, 1)  # Reshape into a column vector
    elseif size(arr, 1) == 1
        return transpose(arr)  # Convert a row vector to a column vector
    end
    return arr
end

# Function to ensure input array is treated as a row array
function ensure_row_array!(arr::Array)
    # If it's a 1D array, reshape it into a row vector
    if ndims(arr) == 1
        return reshape(arr, 1, :)  # Reshape into a row vector
    elseif size(arr, 2) == 1
        return transpose(arr)  # Convert a column vector to a row vector
    end
    return arr
end

# Define a function to print a matrix
function _print(matrix::Matrix; digits=4)
    rows = [eachrow(matrix)...]
    return _print(rows; digits=digits)
end

# Define a function to print a matrix
function _print(vector::Union{Vector{Float64},Vector{Int}}; digits=4)
    vector = ensure_row_array!(vector)

    rows = [eachrow(vector)...]
    return _print(rows; digits=digits)
end

function _print(dict_vector::Vector{<:Dict}; digits=4)
    rows = [[dict for dict in dict_vector]]

    return _print(rows; digits=digits)
end

# Define a function to print an array of rows
function _print(rows::Vector{<:AbstractVector}; digits=4)
    # Determine the max number of lines for any dictionary in the rows
    max_lines_per_dict = maximum([_max_lines(cell) for row in rows for cell in row])

    # Calculate the width of each cell based on the widest content
    num_cols = length(rows[1])
    cell_widths = [
        maximum([_cell_width(row[col]; digits=digits) for row in rows]) for
        col in 1:num_cols
    ]
    overall_width = sum(cell_widths) + length(cell_widths) * 3 + 1

    # Print the top border
    println("+" * "-"^(overall_width - 2) * "+")

    for row in rows
        for i in 1:max_lines_per_dict
            print("|")
            for (j, cell) in enumerate(row)
                print(" ")
                _print(cell, i, cell_widths[j]; digits=digits)
                print(" |")
            end
            println()
        end
        # Print the row separator
        println("+" * "-"^(overall_width - 2) * "+")
    end
end

# Define a helper function to count the number of lines in a dictionary
function _max_lines(cell)
    if cell isa Dict
        return length(keys(cell))
    else
        return 1
    end
end

# Define a helper function to calculate the width of a cell's content
function _cell_width(cell; digits=4)
    if cell isa Dict
        max_key_length = maximum(length(string(k)) for k in keys(cell))
        max_val_length = maximum(
            length(_format_value(v; digits=digits)) for v in values(cell)
        )
        return max_key_length + max_val_length + 2 # for ": " between key and value
    else
        return length(_format_value(cell; digits=digits))
    end
end

# Define a function to print a single cell, either value or dictionary entry
function _print(cell, line::Int, width::Int; digits=4)
    if cell isa Dict
        keys_vals = collect(cell)
        if line <= length(keys_vals)
            key, value = keys_vals[line]
            entry = "$key: $(_format_value(value;digits=digits))"
            print(entry * " "^(width - length(entry)))
        else
            print(" "^width)
        end
    else
        content = _format_value(cell; digits=digits)
        print(content * " "^(width - length(content)))
    end
end

# Helper function to format different types of values
function _format_value(value; digits=4)
    if value isa Float64
        return string(round(value; digits=digits))
    else
        return string(value)
    end
end

# Overload _print for different types
function _print(value::Union{String,Float64,Int}; digits=4)
    return print(_format_value(value; digits=digits))
end

function _print(dict::Dict; digits=4)
    for (key, value) in dict
        println("$key: $(_format_value(value; digits=digits))")
    end
end

function _print(value)
    return print(value)
end

function _print(value::Union{String,Float64,Int})
    return println(value)
end

function print_structure(structure)
    # Iterate over the fields and print their names and values

    fields = fieldnames(typeof(structure))

    for field in fields
        value = getfield(structure, field)
        print("$field: $value")
    end
end

function average_dictionaries(
    arrays::Union{
        Vector{Vector{Dict{String,Float64}}},
        Vector{Matrix{Dict{String,Float64}}},
        Vector{VecOrMat{Dict{String,Float64}}},
    },
)

    # extract the keys from the first dictionary
    key_names = collect(keys(arrays[1][1]))

    # iterate over the dictionaries
    result_array = [zeros(Float64, size(arrays[1])) for _ in 1:length(key_names)]

    for array in arrays
        for (i, key) in enumerate(key_names)
            result_array[i] .+= [dict[key] for dict in array]
        end
    end

    result_array = result_array ./ length(arrays)

    # Initialize the result as a matrix of dictionaries with the same shape as inner arrays
    result = [Dict{String,Float64}() for _ in result_array[1]]

    # Populate the result array with dictionaries
    for idx in eachindex(result_array[1])  # Iterate over each index in the first inner array
        for (j, array) in enumerate(result_array)  # Iterate over each inner array
            result[idx][key_names[j]] = array[idx]  # Create a dictionary entry for each key
        end
    end

    return result
end

# Example
# ================================ MACROS ======================================

export @examine
macro examine(var, value_bool::Bool=true)
    esc_var = esc(var)  # Escape the variable `var` to ensure it's captured correctly
    var_name = string(var)  # Get the variable name as a string

    quote
        println("───────────────────────────────────────")
        println("Variable name: ", $var_name)  # Print the variable name directly
        println("───────────────────────────────────────")

        if $value_bool
            println("Value:")
            _print($esc_var)  # Print the value of the variable
        end

        if isa($esc_var, AbstractArray)
            println("Size: ", size($esc_var))  # Calculate size directly
        end

        println("Type: ", typeof($esc_var))  # Get type directly

        println("───────────────────────────────────────")
    end
end

export @allow_nothing
macro allow_nothing(expr)

    # THE EXISTENCE OF COMMENTS WITHIN THE struct BREAKS THIS MACRO #

    # Ensure the input is a struct definition
    if expr.head != :struct
        error("@allow_nothing must be used with a struct definition")
    end

    # Extract the struct name and fields block
    struct_name = expr.args[2]
    fields_block = expr.args[3]

    # Ensure fields_block is a quoted expression
    if fields_block.head != :block
        error("Unexpected struct definition format")
    end

    # Process each field to include Union{Nothing, ...} or set to Any if no type annotation
    new_fields = map(
        field -> begin
            if field isa Expr && field.head == :(::)  # Check if the field has a type annotation
                field_name = field.args[1]
                field_type = field.args[2]
                :($field_name::Union{Nothing,$field_type})
            elseif field isa LineNumberNode  # Ignore line number nodes (comments)
                nothing
            else  # If no type annotation, set the field type to Any
                :($field::Union{Nothing,Any})
            end
        end, fields_block.args
    )

    # Construct the new struct definition
    return esc(
        quote
            mutable struct $struct_name
                $(new_fields...)
            end
        end,
    )
end
# ==============================================================================

export *

end
