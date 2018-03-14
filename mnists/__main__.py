from . import metadata, prepare


for name in metadata.keys():
    print(name)
    prepare(name)
