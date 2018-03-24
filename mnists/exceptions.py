class IntegrityError(Exception):
    def __init__(self, file, expected_hash, computed_hash):
        self.file = file
        self.expected_hash = expected_hash
        self.computed_hash = computed_hash

    def __str__(self):
        return f"""Failed to validate dataset: {name}
Hash mismatch: {self.computed_hash} should be {self.expected_hash}
Please check your local file for tampering or corruption."""


class UnknownDatasetError(Exception):
    def __init__(self, dataset):
        self.dataset = dataset

    def __str__(self):
        return f"Unknown mnist-like dataset: {dataset}"
