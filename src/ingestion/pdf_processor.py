import fitz  # PyMuPDF
import os
import io
from typing import List, Dict


def extract_text_from_pdf(pdf_path: str) -> List[Dict]:
    """
    Extract text from each page of a PDF.
    Returns a list of dicts with page number and text.
    """
    doc = fitz.open(pdf_path)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()

        if text.strip():
            pages.append({
                "page": page_num + 1,
                "text": text.strip(),
                "type": "text",
                "source": os.path.basename(pdf_path),
            })

    doc.close()
    return pages


def extract_images_from_pdf(pdf_path: str, output_dir: str = "data/temp") -> List[Dict]:
    """
    Extract images from each page of a PDF.
    Returns a list of dicts with page number and image bytes.
    """
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    images = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)

            if base_image:
                image_bytes = base_image["image"]
                # Only keep images larger than 5KB (skip tiny icons)
                if len(image_bytes) > 5000:
                    images.append({
                        "page": page_num + 1,
                        "image_bytes": image_bytes,
                        "type": "image",
                        "source": os.path.basename(pdf_path),
                        "image_index": img_index,
                    })

    doc.close()
    return images


def extract_tables_from_pdf(pdf_path: str) -> List[Dict]:
    """
    Extract tables from PDF using PyMuPDF.
    Returns a list of dicts with page number and table data.
    """
    doc = fitz.open(pdf_path)
    tables = []

    for page_num in range(len(doc)):
        page = doc[page_num]

        # PyMuPDF table extraction
        try:
            page_tables = page.find_tables()
            for table_index, table in enumerate(page_tables):
                table_data = table.extract()

                if table_data and len(table_data) > 1:
                    # Convert table to readable text
                    table_text = _table_to_text(table_data)

                    if table_text.strip():
                        tables.append({
                            "page": page_num + 1,
                            "text": table_text,
                            "type": "table",
                            "source": os.path.basename(pdf_path),
                            "table_index": table_index,
                        })
        except Exception:
            # Skip pages where table extraction fails
            continue

    doc.close()
    return tables


def _table_to_text(table_data: list) -> str:
    """Convert a 2D table list into readable text."""
    if not table_data:
        return ""

    lines = []
    # First row as headers
    headers = [str(cell) if cell else "" for cell in table_data[0]]
    lines.append(" | ".join(headers))
    lines.append("-" * len(lines[0]))

    # Data rows
    for row in table_data[1:]:
        cells = [str(cell) if cell else "" for cell in row]
        lines.append(" | ".join(cells))

    return "\n".join(lines)


def process_pdf(pdf_path: str) -> Dict:
    """
    Full pipeline: extract text, images, and tables from a PDF.
    Returns all extracted content organized by type.
    """
    result = {
        "text_pages": extract_text_from_pdf(pdf_path),
        "images": extract_images_from_pdf(pdf_path),
        "tables": extract_tables_from_pdf(pdf_path),
        "source": os.path.basename(pdf_path),
    }

    total = len(result["text_pages"]) + len(result["images"]) + len(result["tables"])
    result["total_chunks"] = total

    return result