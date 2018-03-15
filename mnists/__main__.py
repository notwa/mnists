from . import metadata, prepare


names = ("train images", "train labels", "test images", "test labels")

for name in metadata.keys():
    # verify every dataset, downloading if necessary.
    # print out the shapes for use in the README.
    print(f" *  `{name}`  ")
    data = prepare(name)
    for name, dat in zip(names, data):
        print(f"    {name} shape: {dat.shape}  ")
    print()
