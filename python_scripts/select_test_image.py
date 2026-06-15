import sys

target_index = int(sys.argv[1])
input_file = "exports/test_inputs_int8.txt"
output_file = "exports/selected_image.txt"

with open(input_file, "r") as f:
    for i, line in enumerate(f):
        if i == target_index:

            values = line.strip().split()

            if len(values) != 784:
                raise ValueError(f"Expected 784 values, got {len(values)}")

            with open(output_file, "w") as out:
                for v in values:
                    out.write(f"{v}\n")   # one hex byte per line

            print(f"Saved image {target_index} → {output_file}")
            break