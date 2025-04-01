from typing import Optional

from smolagents import Tool
from smolagents.models import MessageRole, Model

try:
    from .mdconvert import MarkdownConverter
except:
    from mdconvert import MarkdownConverter
import os
from docling.document_converter import DocumentConverter

class TextInspectorTool(Tool):
    name = "inspect_file_as_text"
    description = """
You cannot load files yourself: instead call this tool to read a file as markdown text and ask questions about it.
This tool handles the following file extensions: [".html", ".htm", ".xlsx", ".pptx", ".wav", ".mp3", ".m4a", ".flac", ".pdf", ".docx"], and all other types of text files. IT DOES NOT HANDLE IMAGES."""

    inputs = {
        "file_path": {
            "description": "The path to the file you want to read as text. Must be a '.something' file, like '.pdf'. If it is an image, use the visualizer tool instead! DO NOT use this tool for an HTML webpage: use the web_search tool instead!",
            "type": "string",
        },
        "question": {
            "description": "[Optional]: Your question, as a natural language sentence. Provide as much context as possible. Do not pass this parameter if you just want to directly return the content of the file.",
            "type": "string",
            "nullable": True,
        },
    }
    output_type = "string"
    md_converter = MarkdownConverter()
    docling_converter = DocumentConverter()

    def __init__(self, model: Model, text_limit: int):
        super().__init__()
        self.model = model
        self.text_limit = text_limit

    def forward_initial_exam_mode(self, file_path, question):
        result = self.md_converter.convert(file_path)

        if file_path[-4:] in [".png", ".jpg"]:
            raise Exception("Cannot use inspect_file_as_text tool with images: use visualizer instead!")

        if ".zip" in file_path:
            return result.text_content

        if not question:
            return result.text_content

        if len(result.text_content) < 4000:
            return "Document content: " + result.text_content

        messages = [
            {
                "role": MessageRole.SYSTEM,
                "content": [
                    {
                        "type": "text",
                        "text": "Here is a file:\n### "
                        + str(result.title)
                        + "\n\n"
                        + result.text_content[: self.text_limit],
                    }
                ],
            },
            {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": "Now please write a short, 5 sentence caption for this document, that could help someone asking this question: "
                        + question
                        + "\n\nDon't answer the question yourself! Just provide useful notes on the document",
                    }
                ],
            },
        ]
        return self.model(messages).content

    def forward(self, file_path, question: Optional[str] = None) -> str:
        DATAPATH=os.getenv("DATAPATH", "")
        file_path = os.path.join(DATAPATH, file_path)
        if not os.path.exists(file_path):
            raise Exception(f"File {file_path} does not exist!")
        
        if file_path[-4:] in [".png", ".jpg"]:
            raise Exception("Cannot use inspect_file_as_text tool with images: use visualizer instead!")
        
        if not any(file_path.endswith(ext) for ext in [".pdf", ".docx", ".xlsx", ".pptx", ".md", ".txt", ".html", ".csv"]):
            result = self.md_converter.convert(file_path)
            doc = result.text_content
        else:
            # use docling to convert the file
            result = self.docling_converter.convert(file_path)
            doc = result.document.export_to_markdown()

        if not question:
            return doc

        messages = [
            {
                "role": MessageRole.SYSTEM,
                "content": [
                    {
                        "type": "text",
                        "text": "You will have to write a short caption for this file, then answer this question:"
                        + question,
                    }
                ],
            },
            {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": "Here is the complete file:\n### "
                        + str(doc)
                        + "\n\n"
                        + doc[: self.text_limit],
                    }
                ],
            },
            {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": "Now answer the question below. Use these three headings: '1. Short answer', '2. Extremely detailed answer', '3. Additional Context on the document and question asked'."
                        + question,
                    }
                ],
            },
        ]
        return self.model(messages).content
    
if __name__ == "__main__":
    tool = TextInspectorTool(model=Model(), text_limit=2000)
    file_path = "e9a2c537-8232-4c3f-85b0-b52de6bcba99.pdf"  
    # file_path = "7bd855d8-463d-4ed5-93ca-5fe35145f733.xlsx"
    file_path = "bec74516-02fc-48dc-b202-55e78d0e17cf.jsonld"
    
    result = tool.forward(file_path)
    print(result)