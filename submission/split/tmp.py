import tarfile
with tarfile.open("submit_validation.tar.gz", "w:gz") as tar:
    tar.add("submit_validation.binproto")