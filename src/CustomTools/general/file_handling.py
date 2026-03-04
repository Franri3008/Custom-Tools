from pathlib import Path
import fitz

def _ensure_dir(path):
    d = Path(path);
    d.mkdir(parents=True, exist_ok=True);
    return d;

def _unique_file(path):
    p = Path(path);
    if not p.exists():
        return p;
    stem = p.stem;
    suffix = p.suffix;
    parent = p.parent;
    i = 1;
    while True:
        np = parent / f"{stem}_{i}{suffix}";
        if not np.exists():
            return np;
        i += 1;

def pdf_to_image(input_path, output_path="./", scale=1, fmt="png"):
    input_path = Path(input_path);
    if not input_path.exists() or input_path.suffix.lower() != ".pdf":
        raise FileNotFoundError("Input PDF not found or invalid path.");
    output_dir = _ensure_dir(output_path);
    saved = [];
    doc = fitz.open(input_path.as_posix());
    for i in range(doc.page_count):
        page = doc.load_page(i);
        mat = fitz.Matrix(scale, scale);
        pix = page.get_pixmap(matrix=mat, alpha=False);
        out_file = _unique_file(output_dir / f"{input_path.stem}_page_{i+1}.{fmt}");
        pix.save(out_file.as_posix());
        saved.append(out_file.as_posix());
    doc.close();
    return saved

def image_to_pdf(input_path, output_path="./"):
    input_path = Path(input_path);
    if not input_path.exists():
        raise FileNotFoundError("Input image not found or invalid path.");
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif"};
    if input_path.suffix.lower() not in exts:
        raise ValueError("Unsupported image format.");
    output_dir = _ensure_dir(output_path);
    out_file = _unique_file(output_dir / f"{input_path.stem}.pdf");
    doc = fitz.open();
    pix = fitz.Pixmap(input_path.as_posix());
    page = doc.new_page(width=pix.width, height=pix.height);
    rect = fitz.Rect(0, 0, pix.width, pix.height);
    page.insert_image(rect, filename=input_path.as_posix());
    doc.save(out_file.as_posix());
    doc.close();
    return out_file.as_posix()

def split_pdf(input_path, output_path="./", ranges=""):
    src_path = Path(input_path);
    if not src_path.exists() or src_path.suffix.lower() != ".pdf":
        raise FileNotFoundError("Input PDF not found.");
    out_dir = _ensure_dir(output_path);
    parts = [];
    with fitz.open(src_path.as_posix()) as doc:
        total = doc.page_count;
        if not ranges:
            for i in range(total):
                out_file = _unique_file(out_dir / f"{src_path.stem}_p{i+1}.pdf");
                nd = fitz.open();
                nd.insert_pdf(doc, from_page=i, to_page=i);
                nd.save(out_file.as_posix());
                nd.close();
                parts.append(out_file.as_posix());
            return parts
        tokens = [t.strip() for t in ranges.split(",") if t.strip()];
        for t in tokens:
            if "-" in t:
                a, b = t.split("-", 1);
                a = int(a) if a else 1;
                b = int(b) if b else total;
            else:
                a = int(t);
                b = a;
            a = max(1, a);
            b = min(total, b);
            if a > b:
                continue;
            nd = fitz.open();
            nd.insert_pdf(doc, from_page=a-1, to_page=b-1);
            out_file = _unique_file(out_dir / f"{src_path.stem}_{a}-{b}.pdf");
            nd.save(out_file.as_posix());
            nd.close();
            parts.append(out_file.as_posix());
    return parts