# src/utils.py
import urllib.request, zipfile, os, pathlib

OCR_URL = "https://ai.stanford.edu/~btaskar/ocr/letter.data.gz"

def fetch_ocr(path="data/raw/letter.data.gz"):
    if pathlib.Path(path).exists():
        return path
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(OCR_URL, path)
    return path