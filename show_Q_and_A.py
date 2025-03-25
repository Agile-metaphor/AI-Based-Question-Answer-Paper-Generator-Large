# A Gradio app that takes a Question Bank TXT and an Answer Bank TXT, generated by question_and_answer_paper_gnerator.py.
# parses them by batch, and displays each question with a show/hide answer toggle.
# Equations are shown in plain text (e.g., "E(t) = 12000 sin(100πt) V/m").

import gradio as gr
import os
import re

def parse_question_bank(qb_text):
    """
    Parse a question bank text into a structure:
    [
      {
        "batch_title": "Batch 1 (Pages 1 - 8 from EM_current_notes.pdf, 714 words, Difficulty: Hard)",
        "questions": [
          {
            "q_number": "1",
            "q_text": "A high-power microwave oven operates at 2.4 GHz ... [15 marks]",
          },
          ...
        ]
      },
      ...
    ]
    """
    lines = qb_text.splitlines()
    results = []
    current_batch_title = None
    current_questions = []
    batch_pattern = re.compile(r'^---\s*Batch\s*(\d+).*---')
    question_pattern = re.compile(r'^[Ff]inalised_Question\s*\(?(\d+)\)?:\s*(.*)$')
    delimiter_pattern = re.compile(r'^\s*={5,}\s*$')
    
    def finish_batch():
        nonlocal current_batch_title, current_questions
        if current_batch_title or current_questions:
            results.append({
                "batch_title": current_batch_title if current_batch_title else "Untitled Batch",
                "questions": current_questions[:]
            })
            current_batch_title = None
            current_questions = []
    
    for line in lines:
        line_stripped = line.strip()
        if batch_pattern.search(line_stripped):
            finish_batch()
            current_batch_title = line_stripped
        elif delimiter_pattern.search(line_stripped):
            finish_batch()
        else:
            qmatch = question_pattern.search(line_stripped)
            if qmatch:
                qnum = qmatch.group(1).strip() if qmatch.group(1) else "?"
                qtxt = qmatch.group(2).strip()
                current_questions.append({
                    "q_number": qnum,
                    "q_text": qtxt
                })
    finish_batch()
    return results

def parse_answer_bank(ab_text):
    """
    Parse an answer bank text into a structure similar to the question bank.
    Returns a list of batches, each with a "batch_title" and list of answers.
    """
    lines = ab_text.splitlines()
    results = []
    current_batch_title = None
    current_answers = []
    batch_pattern = re.compile(r'^---\s*Batch\s*(\d+).*---')
    answer_pattern = re.compile(r'^[Ff]inalised_[Aa]nswer\s*\(?(\d+)\)?:\s*(.*)$')
    delimiter_pattern = re.compile(r'^\s*={5,}\s*$')
    
    def finish_batch():
        nonlocal current_batch_title, current_answers
        if current_batch_title or current_answers:
            results.append({
                "batch_title": current_batch_title if current_batch_title else "Untitled Batch",
                "answers": current_answers[:]
            })
            current_batch_title = None
            current_answers = []
    
    for line in lines:
        line_stripped = line.strip()
        if batch_pattern.search(line_stripped):
            finish_batch()
            current_batch_title = line_stripped
        elif delimiter_pattern.search(line_stripped):
            finish_batch()
        else:
            amatch = answer_pattern.search(line_stripped)
            if amatch:
                anum = amatch.group(1).strip() if amatch.group(1) else "?"
                atxt = amatch.group(2).strip()
                current_answers.append({
                    "a_number": anum,
                    "a_text": atxt
                })
    finish_batch()
    return results

def create_merged_view(q_data, a_data):
    """
    Merge question data with answer data by matching batch numbers and question/answer numbers.
    Returns a consolidated HTML string with each question displayed in a neat format and a show/hide button for the answer.
    """
    batchnum_pattern = re.compile(r'Batch\s+(\d+)')
    
    q_dict = {}
    for b in q_data:
        btitle = b["batch_title"]
        m = batchnum_pattern.search(btitle)
        idx = m.group(1) if m else "?"
        q_dict[idx] = b
    
    a_dict = {}
    for b in a_data:
        btitle = b["batch_title"]
        m = batchnum_pattern.search(btitle)
        idx = m.group(1) if m else "?"
        a_dict[idx] = b
    
    all_keys = sorted(set(q_dict.keys()) | set(a_dict.keys()), key=lambda x: int(x) if x.isdigit() else 999999)
    html = []
    for batch_key in all_keys:
        qb = q_dict.get(batch_key)
        ab = a_dict.get(batch_key)
        if qb:
            batch_title = qb["batch_title"]
        elif ab:
            batch_title = ab["batch_title"]
        else:
            batch_title = f"Batch {batch_key}"
        html.append(f"<h2 style='color: #ff416c;'>{batch_title}</h2>")
        
        questions = qb["questions"] if qb else []
        answers = ab["answers"] if ab else []
        adict_by_num = { a["a_number"]: a["a_text"] for a in answers }
        for qitem in questions:
            qnum = qitem["q_number"]
            qtxt = qitem["q_text"]
            answer_txt = adict_by_num.get(qnum, "<i>No matching answer found.</i>")
            q_html = f"<b>Question ({qnum}):</b> {qtxt}"
            a_html = (
                f"<details style='margin-top:5px;'><summary style='cursor:pointer; color:#ff416c;'>"
                "Show / Hide Answer</summary>"
                f"<p><b>Answer ({qnum}):</b> {answer_txt}</p>"
                "</details>"
            )
            block_html = f"<div style='margin-bottom:15px;'>{q_html}<br>{a_html}</div>"
            html.append(block_html)
    # Wrap everything in a div with inline style to enforce white text.
    return "<div style='color: white;'>" + "\n".join(html) + "</div>"

def display_qna_from_strings(q_text, a_text):
    """
    Given question and answer bank text as strings, parse and merge them into HTML.
    """
    q_data = parse_question_bank(q_text)
    a_data = parse_answer_bank(a_text)
    html_view = create_merged_view(q_data, a_data)
    return html_view

def on_display_qna(q_file, a_file):
    """
    Reads the uploaded question and answer bank files and returns a merged HTML view.
    Handles both dictionary and string returns from the File component.
    """
    if not q_file or not a_file:
        return "<p style='color:red;'>Please upload both Question Bank and Answer Bank files.</p>"
    
    def file_to_text(f):
        if isinstance(f, str):
            with open(f, "r", encoding="utf-8", errors="replace") as fh:
                return fh.read()
        elif isinstance(f, dict) and "data" in f:
            return f["data"].decode("utf-8", errors="replace")
        else:
            return ""
    
    q_text = file_to_text(q_file)
    a_text = file_to_text(a_file)
    return display_qna_from_strings(q_text, a_text)

#### GRADIO UI ####

css_style = """
body { 
    background: #000; 
    color: white; 
}
.gradio-container { 
    background: #000; 
    border-radius: 12px; 
    box-shadow: 0 4px 15px rgba(255, 255, 255, 0.2); 
    padding: 25px; 
    color: white;
}
.gradio-button { 
    background: linear-gradient(45deg, #ff416c, #ff4b2b); 
    color: white; 
    font-weight: bold; 
    border-radius: 8px; 
    border: none; 
    padding: 12px 20px; 
    transition: 0.3s; 
}
.gradio-button:hover { 
    transform: scale(1.05); 
}
h1 { 
    text-align: center; 
    font-size: 2rem; 
    margin-bottom: 10px; 
    color: white;
}
"""

with gr.Blocks(css=css_style) as demo:
    gr.Markdown("# Question & Answer Viewer\n"
                "Upload your **Question Bank** and **Answer Bank** text files.\n"
                "Click **Display Q&A** to see a neatly formatted list of questions with a show/hide answer button.\n"
                "*Equations are displayed in plain text for readability (e.g., E(t) = 12000 sin(100πt) V/m).*")

    with gr.Row():
        question_bank_file = gr.File(label="Question Bank (.txt)", file_types=[".txt"])
        answer_bank_file = gr.File(label="Answer Bank (.txt)", file_types=[".txt"])
    
    display_button = gr.Button("Display Q&A", elem_classes=["gradio-button"])
    output_html = gr.HTML(label="Merged Q&A Output")
    
    display_button.click(
        fn=on_display_qna,
        inputs=[question_bank_file, answer_bank_file],
        outputs=[output_html]
    )

if __name__ == "__main__":
    demo.launch(server_port=7862, share=False)
