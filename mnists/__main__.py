from . import metadata, prepare


headers = ("subdirectory",
           "dataset",
           "train images shape",
           "train labels shape",
           "test images shape",
           "test labels shape")

row = "| {:<20} | {:<20} | {:<20} | {:<20} | {:<20} | {:<20} |"

separators = (":---", ":---", "---:", "---:", "---:", "---:")

urls = {
    "emnist": "//www.nist.gov/itl/iad/image-group/emnist-dataset",
    "fashion-mnist": "//github.com/zalandoresearch/fashion-mnist",
    "mnist": "http://yann.lecun.com/exdb/mnist/",
}

print(row.format(*headers))
print(row.format(*separators))

for name in metadata.keys():
    # verify every dataset, downloading if necessary.
    # print out the shape table for use in the README.
    data = prepare(name)
    prefix = metadata[name][0]
    row_data = [f"[{prefix}][]"]
    row_data += [name.replace("_", "\\_")]
    row_data += [str(array.shape) for array in data]
    print(row.format(*row_data))

print()

for anchor, url in urls.items():
    print(f"[{anchor}]: {url}")
