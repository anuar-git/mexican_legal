
import pdfplumber
from pydantic import BaseModel


class ExtractedPage(BaseModel):
    page_number: int
    raw_text: str
    source_file: str

def extract_pages(pdf_path: str) -> list[ExtractedPage]:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append(ExtractedPage(
                page_number=i,
                raw_text=text,
                source_file=pdf_path,
            ))
    return pages
