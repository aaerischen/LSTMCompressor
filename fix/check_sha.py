import hashlib

path_to_file = "fix\data\enwik5"
path_to_decompressed = "ffix\data\enwik5_decompressed.dat"

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

print(f"SHA256 исходного файла: {sha256(path_to_file)}\n")
print(f"SHA256 файла после декомпрессии: {sha256(path_to_decompressed)}\n")
if sha256(path_to_file) == sha256(path_to_decompressed):
    print("Хеши совпадают, информационная целостность сохранена!")
else:
    print("Хеши не совпадают")