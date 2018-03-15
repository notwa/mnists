from . import metadata, prepare


headers = ("dataset",
           "train images shape",
           "train labels shape",
           "test images shape",
           "test labels shape")

row = "| {:<20} | {:<20} | {:<20} | {:<20} | {:<20} |"

separators = (":---", "---:", "---:", "---:", "---:")

print(row.format(*headers))
print(row.format(*separators))

for name in metadata.keys():
    # verify every dataset, downloading if necessary.
    # print out the shape table for use in the README.
    data = prepare(name)
    row_data = [name.replace("_", "\\_")]
    row_data += [str(array.shape) for array in data]
    print(row.format(*row_data))
