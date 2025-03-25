# Combined Question Generator & Answer Generator Script
# Requires a .env file in the same folder with GROQ_API_KEY defined.

import gradio as gr
import os
import faiss
import numpy as np
import asyncio
import pdfplumber
import pytesseract
import fitz  # PyMuPDF
import re
import time
import math
import random
import json
import nest_asyncio
from PIL import Image
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

# ------------------------- Load environment & setup -------------------------
load_dotenv()  # Load .env in the current folder
nest_asyncio.apply()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY is missing in .env file. Please set it before running.")
client = Groq(api_key=api_key)

# For embeddings if needed (FAISS index).
embedding_model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
dimension = 384
faiss_index = faiss.IndexFlatL2(dimension)

# Global memory for question uniqueness
question_memory = set()

# Global memory for answer uniqueness
answer_memory = set()

# ------------------------- Utility functions -------------------------

def guess_subject_from_pdf(pdf_files):
    """
    If the user hasn't provided a subject name, guess it from the first PDF's file name (no extension).
    E.g. "ElectroMag" from "ElectroMag.pdf".
    If multiple PDFs, just guess from the first one.
    """
    if not pdf_files:
        return "Subject"
    # Take the base name of the first PDF, remove extension
    first_path = pdf_files[0].name
    base = os.path.basename(first_path)
    base_noext = os.path.splitext(base)[0]
    return base_noext

def evaluate_question(question):
    """
    Ensures question uniqueness by tracking a set of seen questions.
    """
    global question_memory
    if question in question_memory:
        return False
    question_memory.add(question)
    return True

def evaluate_answer(answer):
    """
    Ensures answer uniqueness by tracking a set of seen answers.
    """
    global answer_memory
    if answer in answer_memory:
        return False
    answer_memory.add(answer)
    return True

def extract_text_from_pdf_per_page(pdf_file, use_cache=True):
    """
    Extract text from each page of the PDF. If a .json with the same name exists and use_cache=True, use that.
    Otherwise parse with pdfplumber, and if text is too short, fallback to image-based OCR with pytesseract.
    Returns list of (page_text, page_number, doc_name).
    """
    pdf_path = pdf_file.name
    json_path = os.path.splitext(pdf_path)[0] + ".json"
    
    if use_cache and os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as jf:
                data = json.load(jf)
            pages_text = [(item["page_text"], item["page_number"], item["doc_name"]) for item in data]
            return pages_text
        except Exception:
            pass  # Fallback to reading PDF
    
    pages_text = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                # If less than 100 chars, do OCR fallback
                if len(page_text.strip()) < 100:
                    doc = fitz.open(pdf_path)
                    page_fit = doc.load_page(page_num - 1)
                    pix = page_fit.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                    img = Image.open(pix.tobytes("png"))
                    page_text = pytesseract.image_to_string(img)
                    doc.close()
                pages_text.append((page_text, page_num, os.path.basename(pdf_path)))
    except Exception as e:
        raise ValueError(f"Error processing {os.path.basename(pdf_path)}: {str(e)}")
    
    # Save extracted data to JSON for future use
    try:
        json_data = []
        for page_text, page_num, doc_name in pages_text:
            json_data.append({
                "page_text": page_text,
                "page_number": page_num,
                "doc_name": doc_name
            })
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(json_data, jf, ensure_ascii=False, indent=2)
    except Exception:
        pass
    
    return pages_text

# ------------------------- Question Generation -------------------------

def process_batch(batch_text, batch_info, subject, total_batch_words,
                  question_types, randomize_difficulty, fixed_difficulty):
    """
    Splits the total question count into 3 roughly equal parts:
      - Application-oriented
      - Theoretical
      - Mathematical
    Then calls the LLM to generate each portion of questions, combining them at the end.
    """
    # Difficulty
    if randomize_difficulty:
        batch_difficulty = random.choice(["Easy", "Medium", "Hard"])
    else:
        batch_difficulty = fixed_difficulty
    
    # # of questions: 1 per 50 words + 2 extra
    num_questions = math.ceil(total_batch_words / 50) + 2
    
    base_count, remainder = divmod(num_questions, 3)
    application_count = base_count + (1 if remainder > 0 else 0)
    theoretical_count = base_count + (1 if remainder > 1 else 0)
    mathematical_count = base_count
    
    pages_info = ", ".join([f"Page {p} of {doc}" for (_, p, doc) in batch_info])
    
    # Common instructions for math expressions to ensure readable text (no LaTeX).
    eqn_instructions = (
        "When writing mathematical expressions, write them in plain text form, e.g. E(t) = 12000 sin(100πt). "
        "Avoid LaTeX or special symbols that won't be easily readable in a txt file."
    )
    
    # Prompts
    prompt_app = f"""
You are an expert in {subject}. Generate {application_count} unique, high-quality application-oriented questions based on the following study notes (from {pages_info}) with appropriate mark allocation.
Focus on applying theoretical concepts to real-world or practical scenarios. Provide step-based, problem-solving style.
Ensure the questions are suitable for a {batch_difficulty} level exam.
Each question must start with "Finalised_Question(question number):" as a single line.

{eqn_instructions}

Study Notes:
{batch_text}

Format: Separate each question with a double newline.
"""
    prompt_theo = f"""
You are an expert in {subject}. Generate {theoretical_count} unique, high-quality theoretical questions based on the following study notes (from {pages_info}) with appropriate mark allocation.
Focus on conceptual understanding, definitions, and abstract reasoning.
Ensure the questions are suitable for a {batch_difficulty} level exam.
Each question must start with "Finalised_Question(question number):" as a single line.

{eqn_instructions}

Study Notes:
{batch_text}

Format: Separate each question with a double newline.
"""
    prompt_math = f"""
You are an expert in {subject}. Generate {mathematical_count} unique, high-quality mathematical questions based on the following study notes (from {pages_info}) with appropriate mark allocation.
Focus on derivations, numerical analysis, formula-driven tasks, or calculations.
Ensure the questions are suitable for a {batch_difficulty} level exam.
Each question must start with "Finalised_Question(question number):" as a single line.

{eqn_instructions}

Study Notes:
{batch_text}

Format: Separate each question with a double newline.
"""
    # 3 categories
    questions_app, questions_theo, questions_math = [], [], []
    
    try:
        resp_app = client.chat.completions.create(
            model="qwen-2.5-32b",
            messages=[{"role": "user", "content": prompt_app}],
            max_tokens=32768
        )
        generated_text_app = resp_app.choices[0].message.content.strip()
        # Clean up leftover bracket tags
        processed_text_app = re.sub(r'\[(Remember|Understand|Apply|Analyze|Evaluate|Create)\]', '', generated_text_app)
        processed_text_app = re.sub(r'\{(Conceptual Understanding|Critical Thinking|Problem-Solving|Communication|Research Skills)\}', '', processed_text_app)
        processed_text_app = re.sub(r' +', ' ', processed_text_app)
        qa_app = [q.strip() for q in re.split(r'\n\s*\n', processed_text_app) if q.strip()]
        qa_app = [q for q in qa_app if evaluate_question(q)]
        questions_app = qa_app
    except Exception as e:
        questions_app = [f"Error generating application-oriented questions for batch ({pages_info}): {str(e)}"]
    
    try:
        resp_theo = client.chat.completions.create(
            model="qwen-2.5-32b",
            messages=[{"role": "user", "content": prompt_theo}],
            max_tokens=32768
        )
        generated_text_theo = resp_theo.choices[0].message.content.strip()
        processed_text_theo = re.sub(r'\[(Remember|Understand|Apply|Analyze|Evaluate|Create)\]', '', generated_text_theo)
        processed_text_theo = re.sub(r'\{(Conceptual Understanding|Critical Thinking|Problem-Solving|Communication|Research Skills)\}', '', processed_text_theo)
        processed_text_theo = re.sub(r' +', ' ', processed_text_theo)
        qa_theo = [q.strip() for q in re.split(r'\n\s*\n', processed_text_theo) if q.strip()]
        qa_theo = [q for q in qa_theo if evaluate_question(q)]
        questions_theo = qa_theo
    except Exception as e:
        questions_theo = [f"Error generating theoretical questions for batch ({pages_info}): {str(e)}"]
    
    try:
        resp_math = client.chat.completions.create(
            model="qwen-2.5-32b",
            messages=[{"role": "user", "content": prompt_math}],
            max_tokens=32768
        )
        generated_text_math = resp_math.choices[0].message.content.strip()
        processed_text_math = re.sub(r'\[(Remember|Understand|Apply|Analyze|Evaluate|Create)\]', '', generated_text_math)
        processed_text_math = re.sub(r'\{(Conceptual Understanding|Critical Thinking|Problem-Solving|Communication|Research Skills)\}', '', processed_text_math)
        processed_text_math = re.sub(r' +', ' ', processed_text_math)
        qa_math = [q.strip() for q in re.split(r'\n\s*\n', processed_text_math) if q.strip()]
        qa_math = [q for q in qa_math if evaluate_question(q)]
        questions_math = qa_math
    except Exception as e:
        questions_math = [f"Error generating mathematical questions for batch ({pages_info}): {str(e)}"]
    
    final_questions = questions_app + questions_theo + questions_math
    return final_questions, batch_difficulty

def generate_question_bank(pdf_files, subject, fixed_difficulty, randomize_difficulty,
                           total_marks, question_types, slide_threshold, word_threshold,
                           use_cache, set_status_fn=None, set_file_progress_fn=None,
                           set_batch_progress_fn=None, set_total_progress_fn=None):
    """
    Single function to read PDF(s), generate question bank, and save to subject_question_bank.txt
    Uses callbacks (e.g. set_status_fn, set_file_progress_fn, set_batch_progress_fn, set_total_progress_fn) to update progress bars.
    Returns the path of the resulting question bank .txt file.
    """
    # Clear question memory
    global question_memory
    question_memory.clear()
    
    # For total progress: We'll do 40% for question generation overall, 30% for waiting, 30% for answer generation (in total).
    # But let's focus only on question generation for now. We'll handle total progress outside or inside the main generator.

    # Step: processing PDFs
    if set_status_fn:
        set_status_fn("⏳ Starting question generation from PDFs...")

    # Collect all pages from all PDFs
    all_pages = []
    num_files = len(pdf_files)
    for i, pdf_file in enumerate(pdf_files, start=1):
        # Update file progress
        if set_file_progress_fn:
            pct = int((i / num_files) * 100)
            set_file_progress_fn(pct)
        pages = extract_text_from_pdf_per_page(pdf_file, use_cache=use_cache)
        all_pages.extend(pages)
    
    total_slides = len(all_pages)
    # Now create batches
    batches = []
    current_batch = []
    current_batch_word_count = 0
    for page in all_pages:
        text, page_number, doc_name = page
        word_count = len(text.split())
        if word_count >= word_threshold and len(current_batch) == 0:
            batches.append(([page], word_count))
        else:
            current_batch.append(page)
            current_batch_word_count += word_count
            if len(current_batch) >= slide_threshold or current_batch_word_count >= word_threshold:
                batches.append((current_batch, current_batch_word_count))
                current_batch = []
                current_batch_word_count = 0
    if current_batch:
        batches.append((current_batch, current_batch_word_count))

    total_batches = len(batches)
    total_questions_generated = 0
    batch_outputs = []

    # Process each batch
    for batch_index, (batch, batch_words) in enumerate(batches, start=1):
        # Update batch progress
        if set_batch_progress_fn:
            bpct = int((batch_index / total_batches) * 100)
            set_batch_progress_fn(bpct)

        batch_text = "\n\n".join([p[0] for p in batch])
        batch_info = [(p[0], p[1], p[2]) for p in batch]
        questions, batch_difficulty = process_batch(
            batch_text, batch_info, subject, batch_words,
            question_types, randomize_difficulty, fixed_difficulty
        )
        num_batch_questions = len(questions)
        total_questions_generated += num_batch_questions
        
        batch_output = (
            f"--- Batch {batch_index} (Pages {batch[0][1]} - {batch[-1][1]} from {batch[-1][2]}, {batch_words} words, "
            f"Difficulty: {batch_difficulty}) ---\n" + "\n\n".join(questions)
        )
        batch_outputs.append(batch_output)

    avg_questions_per_10 = (total_questions_generated / total_slides) * 10 if total_slides > 0 else 0
    summary = (
        f"Total slides processed: {total_slides}\n"
        f"Total batches: {len(batches)}\n"
        f"Total questions generated: {total_questions_generated}\n"
        f"Average questions per 10 slides: {avg_questions_per_10:.2f}"
    )
    final_output = summary + "\n\n" + ("\n\n=======================================\n\n".join(batch_outputs))

    # Save to txt
    question_bank_file = f"{subject}_question_bank.txt"
    with open(question_bank_file, "w", encoding="utf-8") as f:
        f.write(final_output)

    return question_bank_file

# ------------------------- Answer Generation -------------------------

def extract_batches_from_content(content, delimiter="======================================="):
    """
    Split the .txt question bank by the delimiter to get question batches.
    """
    batches = [segment.strip() for segment in content.split(delimiter) if segment.strip()]
    return batches

def process_batch_answers(batch_text, batch_identifier, subject):
    """
    Prompt model for a set of answers for each question in this batch.
    We want detailed but not overly verbose explanations, with math in plain text.
    """
    # Force plain text for equations
    eqn_instructions = (
        "When writing mathematical expressions, do so in plain text form, e.g. E(t) = 12000 sin(100πt). "
        "Avoid LaTeX or special symbols that won't be easily readable in a txt file."
    )
    prompt = f"""
You are an expert in {subject}. Provide complete, detailed answers for the following questions (from {batch_identifier}), but without extraneous wording or irrelevant tangents.
Make them thorough, step-by-step explanations where needed, in plain text. 
Each answer must start with "Finalised_Answer(question number):" and appear as a single line.

{eqn_instructions}

Questions:
{batch_text}

Format: Separate each answer with a double newline.
"""
    answers = []
    try:
        response = client.chat.completions.create(
            model="qwen-2.5-32b",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=32768
        )
        generated_text = response.choices[0].message.content.strip()
        processed_text = re.sub(r' +', ' ', generated_text)
        possible_answers = [ans.strip() for ans in re.split(r'\n\s*\n', processed_text) if ans.strip()]
        for ans in possible_answers:
            if evaluate_answer(ans):
                answers.append(ans)
    except Exception as e:
        answers = [f"Error generating answers for {batch_identifier}: {str(e)}"]
    return answers

def generate_answer_bank(question_bank_file, subject,
                         set_status_fn=None, set_answer_progress_fn=None, set_total_progress_fn=None):
    """
    Reads the question bank .txt, splits into batches, calls LLM for answers, writes out subject_answer_bank.txt
    Returns the path of the answer bank file.
    """
    global answer_memory
    answer_memory.clear()
    
    if set_status_fn:
        set_status_fn("⏳ Starting answer generation...")

    with open(question_bank_file, "r", encoding="utf-8") as f:
        content = f.read()
    batches = extract_batches_from_content(content)
    total_batches = len(batches)
    batch_outputs = []
    
    for idx, batch_text in enumerate(batches, start=1):
        # Update answer progress
        if set_answer_progress_fn:
            pct = int((idx / total_batches) * 100)
            set_answer_progress_fn(pct)
        batch_identifier = f"Batch {idx} from {os.path.basename(question_bank_file)}"
        answers = process_batch_answers(batch_text, batch_identifier, subject)
        batch_output = f"--- {batch_identifier} ---\n" + "\n\n".join(answers)
        batch_outputs.append(batch_output)

    final_output = "\n\n=======================================\n\n".join(batch_outputs)
    answer_bank_file = f"{subject}_answer_bank.txt"
    with open(answer_bank_file, "w", encoding="utf-8") as f:
        f.write(final_output)

    return answer_bank_file

# ------------------------- Gradio UI (Combined) -------------------------

def full_pipeline(pdf_files,
                  fixed_difficulty, randomize_difficulty, bloom_taxonomy, competency_based,
                  case_study, fill_blanks, total_marks, slide_threshold, word_threshold, use_cache,
                  wait_time,  # user-defined wait (default 30 sec)
                  subject_input):

    # We'll yield updates for status, final output, plus the sub-progress bars:
    #   file_progress, batch_progress, answer_progress, total_progress
    #   total_progress from 0..100 across the entire pipeline.

    # If user didn't set subject, try to guess from first PDF
    subject = subject_input.strip()
    if not subject and pdf_files:
        subject = guess_subject_from_pdf(pdf_files)
    if not subject:
        subject = "Subject"

    # 1) Generate Question Bank
    # We'll consider question generation ~40% of total process
    yield (f"⏳ [0-40%] Starting question generation for {subject}...", 
           "", 0, 0, 0, 0)

    # function to update status text
    def set_status_fn(msg):
        pass  # We'll just rely on the yield states

    # function to update file progress for question gen
    file_progress_internal = 0
    def set_file_progress_fn(value):
        nonlocal file_progress_internal
        file_progress_internal = value

    # function to update batch progress for question gen
    batch_progress_internal = 0
    def set_batch_progress_fn(value):
        nonlocal batch_progress_internal
        batch_progress_internal = value

    # We'll track a "question_gen_progress" from 0..100, then map it to 0..40 of total
    # i.e. total_progress = question_gen_progress * 0.40
    def update_total_progress_qgen():
        # We assume file_progress is 50% of question generation, batch_progress is 50% (approx).
        # or we can just take the average. This is approximate but good enough for demonstration.
        qgen_sub = (file_progress_internal + batch_progress_internal) / 2.0
        total_progress = int(0.40 * qgen_sub)
        return total_progress

    # We'll yield periodically during the question gen
    question_bank_path = generate_question_bank(
        pdf_files=pdf_files,
        subject=subject,
        fixed_difficulty=fixed_difficulty,
        randomize_difficulty=randomize_difficulty,
        total_marks=total_marks,
        question_types={"bloom": bloom_taxonomy, "competency": competency_based,
                        "case_study": case_study, "fill_blanks": fill_blanks},
        slide_threshold=int(slide_threshold),
        word_threshold=int(word_threshold),
        use_cache=use_cache,
        set_status_fn=set_status_fn,
        set_file_progress_fn=set_file_progress_fn,
        set_batch_progress_fn=set_batch_progress_fn
    )
    # Because we didn't yield inside the function, we do a final yield after question generation:
    yield (f"✅ Question bank generated at {question_bank_path}.", 
           "", file_progress_internal, batch_progress_internal, 0, update_total_progress_qgen())

    # 2) Wait for the specified time (30 seconds default)
    yield (f"⏳ [40-70%] Waiting {wait_time} seconds before answer generation...", 
           "", file_progress_internal, batch_progress_internal, 0, 40)
    time.sleep(wait_time)  # wait
    yield (f"✅ Done waiting {wait_time} seconds. Starting answer generation...", 
           "", file_progress_internal, batch_progress_internal, 0, 40)

    # 3) Generate Answers
    # We'll do 30% for answer generation, from total=70..100
    # We'll track a separate answer_progress from 0..100, then map it to 30% of total
    answer_progress_internal = 0
    def set_answer_progress_fn(value):
        nonlocal answer_progress_internal
        answer_progress_internal = value

    def update_total_progress_answer():
        # We consider the entire pipeline up to now is 70%, then answer progress is up to 30% more
        partial = 70 + 0.30 * answer_progress_internal
        return int(partial)

    yield (f"⏳ [70-100%] Generating Answers for {subject}...", 
           "", file_progress_internal, batch_progress_internal, answer_progress_internal, update_total_progress_answer())

    answer_bank_path = generate_answer_bank(
        question_bank_file=question_bank_path,
        subject=subject,
        set_status_fn=set_status_fn,
        set_answer_progress_fn=set_answer_progress_fn
    )

    # final yield
    final_msg = f"✅ All done! Generated {question_bank_path} & {answer_bank_path}"
    final_output = f"Answer bank saved to {answer_bank_path}"
    yield (final_msg, final_output,
           file_progress_internal, batch_progress_internal,
           answer_progress_internal,
           update_total_progress_answer())

# ------------------------- Building the Gradio Interface -------------------------

css_style = """
body { background: linear-gradient(to right, #4A00E0, #8E2DE2); color: white; }
.gradio-container { background: #ffffff; border-radius: 12px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2); padding: 25px; }
.gradio-button { background: linear-gradient(45deg, #ff416c, #ff4b2b); color: white; font-weight: bold; border-radius: 8px; border: none; padding: 12px 20px; transition: 0.3s; }
.gradio-button:hover { transform: scale(1.05); }
.gr-textbox { border: 2px solid #6a11cb; border-radius: 8px; padding: 8px; }
.progress-text { color: #ff416c; font-weight: bold; }
h1 { text-align: center; font-size: 2rem; margin-bottom: 10px; }
h3 { color: #6a11cb; text-align: center; margin-bottom: 20px; }
.category-title { font-weight: bold; margin-top: 10px; color: #6a11cb; }
"""

with gr.Blocks(css=css_style) as demo:
    gr.Markdown("# ✨ Combined Question & Answer Generator\n"
                "**1) Upload one or more PDFs**\n"
                "**2) Customize parameters**\n"
                "**3) Click 'Generate Q&A'**\n"
                "It will:\n"
                "- Generate a question bank (in ~33% application, 33% theoretical, 33% math style)\n"
                "- Wait 30 seconds\n"
                "- Then generate detailed answers to those questions.\n"
                "Outputs are saved to `<subject>_question_bank.txt` and `<subject>_answer_bank.txt`.\n\n"
                "Mathematical expressions are formatted in plain text, e.g. `E(t) = 12000 sin(100πt)`. "
                "Excessive LaTeX is avoided to keep them readable in .txt files."
               )

    with gr.Row():
        pdf_files = gr.Files(label="📄 Upload PDF(s) for question generation", file_types=[".pdf"])

    with gr.Row():
        subject_input = gr.Textbox(label="📚 Subject (if empty, uses first PDF name)", placeholder="")

    with gr.Row():
        with gr.Column():
            fixed_difficulty = gr.Dropdown(label="🎯 Fixed Difficulty Level", choices=["Easy", "Medium", "Hard"], value="Medium")
            randomize_difficulty = gr.Checkbox(label="Randomize Difficulty Per Batch", value=True)
            total_marks = gr.Slider(label="📊 Total Marks", minimum=50, maximum=2000, value=100, step=50)
            slide_threshold = gr.Number(label="Batch Slide Threshold", value=22)
            word_threshold = gr.Number(label="Batch Word Threshold", value=4500)
            wait_time = gr.Number(label="Wait Time (seconds) before answer generation", value=30)
        with gr.Column():
            use_cache = gr.Checkbox(label="Use Cached JSON Data (if available)", value=True)
            bloom_taxonomy = gr.Checkbox(label="Use Bloom's Taxonomy (internal)", value=True)
            competency_based = gr.Checkbox(label="Use Competency-Based Categories (internal)", value=True)
            case_study = gr.Checkbox(label="Include Case Study Questions", value=False)
            fill_blanks = gr.Checkbox(label="Include Fill-in-the-Blanks", value=False)

    # 6 progress bars / status boxes:
    # 1) Status
    # 2) Final output
    # 3) File progress (question generation)
    # 4) Batch progress (question generation)
    # 5) Answer generation progress
    # 6) Total progress
    status = gr.Textbox(label="🚀 Status", value="", interactive=False)
    final_output = gr.Textbox(label="Final Output", value="", lines=4, interactive=False)
    file_progress = gr.Slider(label="File Processing Progress (Q-Gen)", minimum=0, maximum=100, value=0, interactive=False)
    batch_progress = gr.Slider(label="Batch Generation Progress (Q-Gen)", minimum=0, maximum=100, value=0, interactive=False)
    answer_progress = gr.Slider(label="Answer Generation Progress", minimum=0, maximum=100, value=0, interactive=False)
    total_progress = gr.Slider(label="Total Progress", minimum=0, maximum=100, value=0, interactive=False)

    generate_button = gr.Button("⚡ Generate Q&A", elem_classes=["gradio-button"])

    def main_pipeline(pdf_files, subject_input, fixed_difficulty, randomize_difficulty,
                      bloom_taxonomy, competency_based, case_study, fill_blanks, total_marks,
                      slide_threshold, word_threshold, use_cache, wait_time):
        """
        Generator function that yields multiple statuses and updates for progress bars.
        We have to yield exactly the 6 outputs in the correct order: status, final_output, file_progress,
        batch_progress, answer_progress, total_progress
        """
        # We'll use the "full_pipeline" function from above, but each yield we map to these 6 outputs
        pipe = full_pipeline(pdf_files, fixed_difficulty, randomize_difficulty,
                             bloom_taxonomy, competency_based, case_study, fill_blanks,
                             total_marks, slide_threshold, word_threshold, use_cache,
                             wait_time, subject_input)
        for (msg, out, fp, bp, ap, tp) in pipe:
            yield (msg, out, fp, bp, ap, tp)

    generate_button.click(
        main_pipeline,
        inputs=[
            pdf_files, subject_input,
            fixed_difficulty, randomize_difficulty,
            bloom_taxonomy, competency_based, case_study, fill_blanks,
            total_marks, slide_threshold, word_threshold,
            use_cache, wait_time
        ],
        outputs=[status, final_output, file_progress, batch_progress, answer_progress, total_progress]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
