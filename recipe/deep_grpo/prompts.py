"""DEEP-GRPO teacher prompts.

Templates for teacher-guided suffix synthesis: the teacher is shown a failed
student trajectory and asked to keep the longest safe prefix character-for-
character, then continue with a clean correct solution. The V2 variant is
prefix-aware: it treats an already-verified partial solution embedded in the
context as immutable background (used for deeper forest nodes).
"""

TEACHER_SELECTION_PROMPT_TEMPLATE = """
You are an expert Reasoning and Error Analysis Agent.
I will provide you with a **Problem**, a **Reference Solution** (Standard Answer), and a **Student's Incorrect Trajectory**.

Your task is to compare the Student's Trajectory against the Reference Solution and identify the **FIRST step** where the student made a mistake.
The mistake could be a logical error, a calculation error, a wrong code implementation, or a hallucination.

### Input Data
**Problem**:
{instruction}

**Reference Solution**:
{reference}

**Student Trajectory**:
{steps}

### Instructions
1. **Analyze Step-by-Step**: 
   - Go through the student's steps one by one.
   - Explicitly state whether each step is correct or incorrect and why.
2. **Identify the First Error**:
   - Locate the exact index (0-based) of the first step that deviates from the correct logic.
3. **Output JSON**:
   - Finally, output the result in a JSON object containing the key "first_error_step_index".

### Output Format
First, provide your analysis. Then, output the JSON in the following format:

### Analysis
[Your step-by-step reasoning here...]

### Result
```json
{{
    "first_error_step_index": <int>
}}
```

(If the very first step is wrong, return 0. If no error is found, return -1.)

Now, start your analysis.
"""


TEACHER_SUFFIX_SYNTHESIS_PROMPT_TEMPLATE_V2 = """You are an expert problem solver, programmer, and error analyst.

I will provide you with a **Context** and a **Student's Failed Trajectory**.

**Important note about the Context**: The Context contains the problem statement, and may additionally contain a **previously-verified correct partial solution** that has already been checked for correctness. Treat the entire Context as given background. DO NOT re-analyze, modify, or include any part of the Context in your output.

The **Student's Failed Trajectory** is the new portion of work that picks up from where the Context ends. Your job is to analyze only the Student's Failed Trajectory.

Your task is to:
1. Solve the problem independently from the Context.
2. Identify the **FIRST** point in the Student's Failed Trajectory where the trajectory becomes wrong, misleading, unsupported, ambiguous, or would need to be contradicted later.
3. Output a clean correction that:
   - Preserves the longest safe prefix of the Student's Failed Trajectory **exactly character-for-character**, but only up to (not including) the first unsafe token
   - Then continues with your own correct reasoning to a complete final answer
   - Never repairs, contradicts, reinterprets, or apologizes for any copied prefix text

### Context
{instruction}

### Student's Failed Trajectory
{student_response}

### Critical Instructions
- Maximize exact overlap with the Student's Failed Trajectory **only while the copied prefix remains fully safe and correct**.
- Keep the safe prefix **EXACTLY** as-is: same formatting, same symbols, same whitespace, same tokens.
- Do NOT copy any statement, variable binding, calculation, operator, code token, API choice, or conclusion that is wrong, misleading, incomplete in a harmful way, or later needs correction.
- If the student's first token is already unsafe, copy nothing from the Student's Failed Trajectory and start directly with a clean solution.
- From the first unsafe token onward, replace the rest with your own correct solution.
- Do NOT use repair language such as "Wait", "Actually", "Correction", "I made a mistake", or "Let's start over".
- The copied prefix plus your continuation must be monotonic: the continuation must never contradict or revise anything in the copied prefix.
- DO NOT include or reproduce any content from the Context. Only reproduce from the Student's Failed Trajectory.
- Your output should be complete and self-contained from the point where the Student's Failed Trajectory begins.

### Output Format
Use the EXACT section headers below. Write the solution as plain text (NOT inside JSON or code blocks).

[ERROR_DESCRIPTION]
<brief description of the first unsafe point in the Student's Failed Trajectory>

[CORRECT_SOLUTION]
<the longest safe exact prefix from the Student's Failed Trajectory, followed by your clean continuation to the final answer>

Now, start your analysis.
"""


TEACHER_SUFFIX_SYNTHESIS_PROMPT_TEMPLATE = """You are an expert problem solver, programmer, and error analyst.

I will provide you with a **Problem** and a **Student's Failed Trajectory**.

Your task is to:
1. Solve the problem independently.
2. Identify the **FIRST** point where the Student's Failed Trajectory becomes wrong, misleading, unsupported, ambiguous, or would need to be contradicted later.
3. Output a **COMPLETE clean solution** that preserves the longest safe prefix of the student's work **exactly character-for-character** up to (but not including) the first unsafe token, then continues with your own correct solution.

### Problem
{instruction}

### Student's Failed Trajectory
{student_response}

### Critical Instructions
- Maximize exact overlap with the Student's Failed Trajectory **only while the copied prefix remains fully safe and correct**.
- Keep the safe prefix **EXACTLY** as-is: same formatting, same symbols, same whitespace, same tokens.
- Do NOT copy any statement, variable binding, calculation, operator, code token, API choice, or conclusion that is wrong, misleading, incomplete in a harmful way, or later needs correction.
- If the student's first token is already unsafe, copy nothing from the Student's Failed Trajectory and start directly with a clean solution.
- From the first unsafe token onward, replace the rest with your own correct solution.
- Do NOT use repair language such as "Wait", "Actually", "Correction", "I made a mistake", or "Let's start over".
- The copied prefix plus your continuation must be monotonic: the continuation must never contradict or revise anything in the copied prefix.
- Your solution should be complete and self-contained.

### Output Format
Use the EXACT section headers below. Write the solution as plain text (NOT inside JSON or code blocks).

[ERROR_DESCRIPTION]
<brief description of the first unsafe point>

[CORRECT_SOLUTION]
<the longest safe exact prefix from the Student's Failed Trajectory, followed by your clean continuation to the final answer>

Now, start your analysis.
"""
