"""Download official GloVe 6B and the assignment's BERT checkpoint."""
import argparse
import json
import struct
import time
import urllib.request
import zipfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BERT_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"


def download_zip_member(url, name, target):
    """Download one official ZIP member via HTTP Range, checking size and CRC."""
    target = Path(target)
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as response:
        total = int(response.headers["Content-Length"])

    def get_range(start, end):
        response = urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"}), timeout=120)
        if response.status != 206:
            response.close()
            raise IOError("The ZIP server did not honor the HTTP Range request.")
        return response

    tail_start = max(0, total - 65536)
    with get_range(tail_start, total-1) as response:
        tail = response.read()
    end = tail.rfind(b"PK\x05\x06")
    if end < 0:
        raise IOError("Missing ZIP end-of-central-directory record.")
    eocd = struct.unpack("<4s4H2LH", tail[end:end+22])
    central_offset, central_size = eocd[6], eocd[5]
    if central_offset >= tail_start:
        central = tail[central_offset-tail_start:central_offset-tail_start+central_size]
    else:
        with get_range(central_offset, central_offset+central_size-1) as response:
            central = response.read()
    cursor, member = 0, None
    while cursor < len(central):
        entry = struct.unpack("<4s6H3L5H2L", central[cursor:cursor+46])
        n, x, c = entry[10:13]
        filename = central[cursor+46:cursor+46+n].decode("utf-8")
        if filename == name:
            member = entry
            break
        cursor += 46+n+x+c
    if member is None or member[4] != 8:
        raise IOError("Expected deflated GloVe member was not found.")
    crc_expected, compressed, uncompressed, offset = member[7], member[8], member[9], member[16]
    with get_range(offset, offset+29) as response:
        header = struct.unpack("<4s5H3L2H", response.read())
    start = offset+30+header[9]+header[10]
    # Reuse a contiguous prefix from an earlier full-archive download.
    prefix = target.parent / "glove.6B.zip.part"
    compressed_file = target.with_suffix(".deflate.part")
    have = compressed_file.stat().st_size if compressed_file.exists() else 0
    if have == 0 and prefix.exists() and prefix.stat().st_size > start:
        reusable = min(compressed, prefix.stat().st_size-start)
        with prefix.open("rb") as source, compressed_file.open("wb") as destination:
            source.seek(start)
            remaining = reusable
            while remaining:
                block = source.read(min(1024*1024, remaining))
                if not block:
                    raise IOError("Truncated archive prefix.")
                destination.write(block)
                remaining -= len(block)
        have = reusable
    if have < compressed:
        with get_range(start+have, start+compressed-1) as response, compressed_file.open("ab") as destination:
            last = time.monotonic()
            while block := response.read(1024*1024):
                destination.write(block)
                have += len(block)
                if time.monotonic()-last > 10:
                    print(f"{name}: {have/1e6:.1f}/{compressed/1e6:.1f} MB compressed", flush=True)
                    last = time.monotonic()
    if have != compressed:
        raise IOError("Compressed GloVe member has the wrong size.")
    inflater, crc, count = zlib.decompressobj(-15), 0, 0
    temporary = target.with_suffix(".tmp")
    with compressed_file.open("rb") as source, temporary.open("wb") as destination:
        while block := source.read(1024*1024):
            expanded = inflater.decompress(block)
            destination.write(expanded)
            count += len(expanded)
            crc = zlib.crc32(expanded, crc)
        expanded = inflater.flush()
        destination.write(expanded)
        count += len(expanded)
        crc = zlib.crc32(expanded, crc)
    if not inflater.eof or count != uncompressed or crc != crc_expected:
        raise IOError("GloVe size/CRC verification failed; retry without partial downloads.")
    temporary.replace(target)
    compressed_file.unlink()
    print(f"Extracted and verified {name}: {count} bytes", flush=True)


def download(url, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print(f"Already downloaded: {target}", flush=True)
        return
    partial = target.with_suffix(target.suffix + ".part")
    # Restart rather than append when the server does not honor Range.
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
    with urllib.request.urlopen(request, timeout=120) as response:
        append = offset > 0 and response.status == 206
        if not append:
            offset = 0
        total = int(response.headers.get("Content-Length", 0)) + offset
        last = time.monotonic()
        with partial.open("ab" if append else "wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
                offset += len(chunk)
                if time.monotonic() - last > 10:
                    print(f"{target.name}: {offset / 1e6:.1f}/{total / 1e6:.1f} MB", flush=True)
                    last = time.monotonic()
        if total and offset != total:
            raise IOError(f"Incomplete download: expected {total}, got {offset}")
    partial.replace(target)
    print(f"Downloaded: {target.name}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("resource", choices=["glove", "bert", "all"], default="all", nargs="?")
    args = parser.parse_args()
    if args.resource in {"glove", "all"}:
        text = ROOT / "data/glove.6B.100d.txt"
        if not text.exists():
            download_zip_member("https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip", "glove.6B.100d.txt", text)
    if args.resource in {"bert", "all"}:
        folder = ROOT / "models/bert-base-uncased"
        base = f"https://huggingface.co/google-bert/bert-base-uncased/resolve/{BERT_REVISION}"
        for filename in ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt", "model.safetensors"]:
            download(f"{base}/{filename}", folder / filename)
        (folder / "source.json").write_text(json.dumps({"model": "google-bert/bert-base-uncased", "revision": BERT_REVISION}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
