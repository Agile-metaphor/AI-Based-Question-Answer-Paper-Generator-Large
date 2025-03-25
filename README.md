
Original [
**Enhanced Question Bank Generator**
An AI-powered application that automatically generates educational question banks from PDF documents, leveraging vector embeddings and large language models.
About
The Enhanced Question Bank Generator is a sophisticated tool designed for educators and content creators who need to quickly generate high-quality assessment materials from existing documents. This application combines the power of natural language processing, vector embeddings, and large language models to analyze PDF content and create relevant, well-structured questions with appropriate mark allocation. Whether you're preparing for classroom assessments, creating practice materials, or developing standardized tests, this tool streamlines the process while ensuring pedagogical soundness.

**Features**
The application offers a comprehensive suite of features designed to create effective educational assessments. It processes PDF documents with advanced text extraction capabilities, including OCR support for scanned materials. Using vector embeddings and similarity search, it identifies key concepts and relevant content sections to inform question generation. Users can customize their question banks by specifying subject matter, difficulty levels, and distribution between multiple-choice and subjective questions. The system incorporates established educational frameworks like Bloom's Taxonomy and competency-based categories to ensure questions target appropriate cognitive levels and skills. Additionally, it implements a sophisticated mark allocation algorithm that distributes points fairly based on question complexity and type.

**Requirements**
To run the Enhanced Question Bank Generator, you'll need Python 3.8 or higher installed on your system. The application requires a Groq API key for accessing the large language model used in question generation. For processing scanned PDF documents, Tesseract OCR is recommended but optional. All other dependencies are handled through the provided requirements file, including libraries for vector embedding, similarity search, PDF processing, and the web interface.
]

(Note: Use wsl and conda if on windows, saves stress)

Improvements(-to be rewritten and explained by hand later):
Designed to handle large pdfs, splits them into batches(words/pages) and generates questions and answers based on content every batch(words/pages) of content.

Safety:
Groq api keys are now stored in a .env file as opposed to plain text

Combined Q&A Pipeline:
qagp.py now not only generates a question bank but also produces a corresponding answer bank automatically, whereas the original only created a question bank.
Balanced Question Categories:
Questions are split into approximately 33% application‐oriented, 33% theoretical, and 33% mathematical questions—providing a more balanced and comprehensive assessment.

Detailed Answer Generation:
The pipeline generates detailed, step‐by‐step explanations for each question. The answers are formatted in plain text (textbook style) with equations such as “E(t) = 12000 sin(100πt)” for clear readability.
Configurable Wait Time:
A user-set wait time (default 30 seconds) is introduced between the question generation phase and the answer generation phase to manage processing flow and ensure proper sequencing.

Progress Monitoring:
Multiple progress bars are implemented:
File Processing Progress for question generation.
Batch Generation Progress for creating question batches.
Answer Generation Progress for the answer bank.
Total Progress reflecting the entire pipeline status.

User-Adjustable Parameters:
All key variables (difficulty, total marks, question counts, slide and word thresholds, wait time, etc.) are now configurable via the Gradio interface.
Subject Auto-Assignment & Output Naming:
If the user leaves the subject blank, it automatically defaults to the first PDF’s filename.
Output files are saved with filenames that append “_question_bank.txt” and “_answer_bank.txt” to the subject name.

Enhanced Prompt Engineering:
Ensure formatting, mark allocation, and plain-text representation of mathematical equations, avoiding LaTeX that might be hard to read in a text file.

Integrated Q&A Viewer:
The separate show_qna.py file offers a clean interface to load and view the question and answer banks side by side, with each question displayed neatly and a show/hide button for its corresponding answer, organized by batch.

Improved Text Extraction & Caching:
Text extraction from PDFs is provided with OCR fallback using PyMuPDF and pytesseract.
The system supports caching of extracted text (via JSON files) to speed up subsequent processing.

Interface for display:
The Gradio UI is modernized with customizable CSS, ensuring a clean visual experience (and, in the case of show_qna.py, a dark background with contrasting text, if desired).
Overall, the qagp.py and show_qna.py combination offers a full, customizable end-to-end exam content generator and viewer that goes far beyond the basic functionality of the original questionpapergenerator.py.








