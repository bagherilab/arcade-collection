import io
import json
import tarfile


def build_tar_instance(contents):
    buffer = io.BytesIO()

    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for file_key, content in contents.items():
            byte_array = json.dumps(content).encode("utf-8")
            info = tarfile.TarInfo(file_key)
            info.size = len(byte_array)
            tar.addfile(info, io.BytesIO(byte_array))

    return tarfile.open(fileobj=io.BytesIO(buffer.getvalue()), mode="r")
