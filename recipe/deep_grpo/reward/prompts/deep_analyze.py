ANSWER_CORRECTNESS_EVALUATE_PROMPT = """
You are an automatic grader for data science questions.

Your goal is to grade whether the MODEL ANSWER is correct or incorrect when compared to the GROUND TRUTH, given the QUESTION.

You must be strict and binary: the answer is either correct (1) or not (0).

Focus on semantic content, not wording or formatting.

### DATA SCIENCE CONSTRAINTS

- Numerical Equivalence: Treat mathematically equal numbers as a MATCH (e.g., 0.5 = 50% = 1/2). Allow reasonable rounding unless the QUESTION specifies precision.

- Collections: If the QUESTION does not strictly ask for "ranking" or "order", treat lists as Sets (order does NOT matter).

### PATCH (important clarification about short answers)

- The QUESTION already defines the constraints/conditions. Therefore, the MODEL ANSWER is allowed to be a short final output (e.g., a single number, a single entity name, yes/no) without restating the conditions, **as long as it is unambiguous under the QUESTION context** and matches the required ground-truth content.

- Do NOT penalize the MODEL ANSWER solely for omitting constraints that are already fully specified by the QUESTION. Only penalize if the omission makes the answer ambiguous or changes the meaning.

------------------

[QUESTION]

{question}

[GROUND TRUTH]

{ground_truth}

[MODEL ANSWER]

{model_answer}

------------------

Scoring rubric (0/1):

- **1 (Correct)** = The answer is fully correct and semantically equivalent to the ground truth. It contains all the same facts and satisfies all constraints. Only non-material differences (like wording, formatting, or harmless grammar) are permitted.
- **0 (Incorrect)** = The answer is anything less than fully correct. This includes answers that are mostly correct but have minor inaccuracies, miss important conditions, contain any wrong facts, or are incomplete.

Now evaluate in two steps.

Step 1 – Structured comparison (mandatory):

Write a concise but explicit analysis with the following sections:

A) Key facts in GROUND TRUTH:
- List the atomic facts/constraints (who/what, value(s), conditions/filters, quantity/aggregation if any).

B) Key facts in MODEL ANSWER:
### PATCH (interpret short answers in context)
- If the MODEL ANSWER is short (e.g., "5"), interpret it as: "the requested value under the QUESTION constraints is 5", and extract the implied factual content accordingly.
- List the atomic facts/constraints the model answer asserts (explicitly or implicitly under the QUESTION).

C) Comparison:
- List MATCHES (facts/constraints that align).
- List DIFFERENCES. **Any difference, unless it is purely non-material (e.g., wording), will result in a score of 0.**
- Briefly justify why any differences are material or non-material.

Do NOT output the score in this step.

Step 2 – Final score:

On a new line, output the score strictly in this format:

### Final Score: <0|1>

The last line of your entire response must contain only the final score in this exact format, with nothing else on that line.
"""

# ANSWER_CORRECTNESS_EVALUATE_PROMPT = """
# You are a strict data science grader.
# Grade the <model_answer> against the <ground_truth> based on the <question>.

# ### GRADING RULES
# 1. **Truthfulness**: The answer must match the semantic meaning of the Ground Truth.
# 2. **Flexibility**: Allow numerical equivalence (0.5 = 50%), reasonable rounding (0.333 ≈ 0.33), and contextually valid short answers.
# 3. **Strictness**: The answer is INCORRECT (Score 0) if it contains *any* false information or hallucinations, even if the correct answer is also present.

# ### DATA
# <question>
# {question}
# </question>

# <ground_truth>
# {ground_truth}
# </ground_truth>

# <model_answer>
# {model_answer}
# </model_answer>

# ### OUTPUT FORMAT
# 1. **Reasoning**: A concise explanation (1 sentence) verifying the match and checking for hallucinations.
# 2. **Score**: On a new line, output strictly: SCORE: <0|1>
# """



RESPONSE_QUALITY_PROMPT = """
You are a data science evaluation assistant.

Your task is to evaluate the quality of the MODEL RESPONSE's reasoning process for the given data science QUESTION.

The MODEL RESPONSE may come in different trace styles:

(A) Agent-style multi-step traces with multiple tagged blocks such as
    <Understand>, <Analyze>, <Code>, <Execute>, and a final <Answer>.
(B) Reasoning-style traces where there is ONLY ONE <Analyze> block that contains almost all reasoning,
    followed by a final <Answer>.
(C) Minimal/partial traces where tags may be missing or incomplete; you should still evaluate the reasoning
    based on the available content.

The <Execute> blocks represent raw environment outputs (e.g., results of running code). They are NOT themselves
reasoning steps, but good reasoning should correctly interpret, verify, and use them.

Your goal is to judge how good the whole reasoning trace is for solving the QUESTION.
Use the GROUND TRUTH only as a reference for what the correct final answer should be.
Do NOT re-grade answer correctness here; overall correctness is handled by a separate metric.
A correct final <Answer> does NOT automatically imply a high reasoning-quality score, and an incorrect final
<Answer> does NOT automatically imply a low reasoning-quality score. You are grading the reasoning process itself.

------------------
[QUESTION]
{question}

[GROUND TRUTH]
{ground_truth}

[MODEL RESPONSE]
{model_response}
------------------

Scoring (high level):
Assign a single integer score from 0 to 4 for the quality of the reasoning process:

- 0: Very poor – reasoning is mostly wrong, missing, or incoherent.
- 1: Weak – some relevant ideas, but many errors, gaps, or unclear steps.
- 2: overall direction is reasonable. However, the solution may be over-engineered (e.g., unnecessary complexity/defensiveness), relies on monolithic code blocks (doing too much in one step), or uses inefficient "rewrite-everything" debugging strategies.
- 3: Strong – reasoning is mostly correct and coherent, and code execution is generally smooth, with only minor issues.
- 4: Exceptional – reasoning is clear, careful, and technically sound. The code is concise, modular, and readable. The model handles errors surgically (fixing only the specific line) rather than over-correcting.

**CRITICAL FAILURE CHECK (ZERO TOLERANCE POLICY):**
- **Language Integrity**: You must strictly inspect the language quality. If the report displays **gibberish (meaningless characters)**, **unjustified mixed languages** (e.g., switching between English, Chinese, and random tokens mid-sentence), or **repetitive loops of text**, this is a critical failure.
- **TOTAL PENALTY**: If such behavior is detected, the report is considered a complete failure. You MUST assign a score of **0**, regardless of any partial content.

Now perform the evaluation in two steps.

============================================================
Step 1 – Analysis (detailed, step by step)
============================================================

Carefully read the entire MODEL RESPONSE (including all tags and the final <Answer>) and write out your reasoning
about its reasoning quality.

IMPORTANT: Do NOT mention the word "SCORE" anywhere in this analysis section.

You MUST do the following:

0) **SAFETY CHECK (Language Integrity)**:
   - Before analyzing specific steps, scan the response.
   - If it contains **gibberish, chaotic mixed languages, or infinite loops**, STOP the detailed step-by-step analysis.
   - Simply write: "CRITICAL FAILURE: The response is unintelligible/gibberish."
   - Then skip directly to Step 2 to output SCORE: 0.

1) Identify and enumerate the reasoning steps in order, then comment on EACH step.
   - If the MODEL RESPONSE contains multiple tagged reasoning blocks (e.g., <Understand>, <Analyze>, <Code>),
     treat EACH reasoning-related tagged block as a separate step, in order of appearance.
     Note: <Execute> blocks are NOT reasoning steps; use them only as evidence to verify or critique nearby reasoning steps.
   - If the MODEL RESPONSE contains ONLY ONE <Analyze> (Reasoning-style), you MUST split that <Analyze> into multiple
     logical sub-steps (at least 2 when feasible), based on natural reasoning breaks (topic shifts, decisions,
     computations, evidence interpretation, etc.). Then comment on each sub-step in order.
   - If tags are missing or incomplete, infer steps by natural paragraph/sentence boundaries and major action transitions, and evaluate them in order.

   For EACH step (or sub-step), you MUST follow ALL sub-steps below, in order:
   a) Explain in your own words what this step is doing.
   b) Explain whether it is logically connected to the previous step.
   c) Check any related <Execute> outputs (if any) and state whether this step uses/interprets those outputs correctly.
      - If this step SHOULD have used relevant <Execute> evidence but ignores it, explicitly note that as a weakness.
      - If the step cites numbers/results not supported by <Execute>, explicitly flag it.
   d) State whether this step is reasonable/correct/helpful or problematic, and point out specific issues if present.

2) After going through all steps, evaluate the overall reasoning trace as a whole:
   - Does the sequence of steps logically lead toward answering the QUESTION?
   - Is it internally consistent (assumptions, definitions, variable usage, units, data subsets, etc.)?
   - Is the approach appropriate for the task (not overcomplicated, not missing key methodology)?
   - Does it demonstrate good analytical hygiene (sanity checks, robustness considerations, clear mapping from evidence to claims)?
   - If code is present: is the code plan sensible, are results interpreted properly, are there obvious methodological pitfalls?
   - Is the execution efficient? Does the model write correct code on the first try, or does it get stuck in loops of "code error -> fix -> error"? (Penalize excessive brute-force debugging).
   - Is the code granularity appropriate? Does the model break tasks into small, verifiable chunks (Good), or does it attempt to run a huge monolithic script that fails entirely upon a single syntax error (Bad)?
   - Is the solution proportional to the problem? Check for "Performative Rigor." For example, writing a complex file-logging system just to see data summary statistics is a failure of judgment. Penalize approaches that are defensively complex/verbose without adding actual value.


============================================================
Step 2 – Final score
============================================================

After you finish your analysis, on a new line output only the final score in this exact format:

SCORE: <0 or 1 or 2 or 3 or 4>

The last line of your entire response must contain only this score line, with nothing else on that line.
"""

REPORT_QUALITY_EVALUATE_PROMPT = """
You are a data science report quality evaluator.

You are given:
- a user instruction describing a data analysis task, and
- a generated data science report (the final output) written in response to that instruction.

**Context & Constraints:**
- You are evaluating the **final report content only**. The user did NOT provide the intermediate code execution steps or raw data.
- **Do NOT penalize the report for not showing code**.
- **Truthfulness Assumption**: Assume the specific numbers/results presented are calculated correctly from the data.
- Your goal is to judge whether this report constitutes a **high-quality, professional answer** to the user's business or technical question.

**CRITICAL FAILURE CHECK (ZERO TOLERANCE POLICY):**
- **Language Integrity**: You must strictly inspect the language quality. If the report displays **gibberish (meaningless characters)**, **unjustified mixed languages** (e.g., switching between English, Chinese, and random tokens mid-sentence without context), or **repetitive loops of text**, this is a critical failure.
- **TOTAL PENALTY**: If such behavior is detected, the report is considered a complete failure. You MUST assign a score of **0** for **ALL FIVE DIMENSIONS** (Task Fulfillment, Richness, Soundness, Visualization, Readability), regardless of any partial content.

Your task is to evaluate the quality of the report along five dimensions.

--------------------
[INSTRUCTION]
{instruction}

[GENERATED REPORT]
{report}
--------------------

For each of the following five dimensions, assign an **integer score from 0 (very poor) to 4 (excellent)**.
Be strict but fair: a score of 4 should be reserved for truly outstanding, executive-ready reports.

---

### 1. Task Fulfillment (Did it answer the specific questions asked?)

**Definition**: How well does the report directly address the specific goals in the instruction?

**Key Evaluation Criteria**:
- **Completeness**: Did it answer **ALL** parts of the user's prompt?
- **Directness**: Does it provide a clear, direct answer to the question posed in the instruction?
- **Actionability**: Are the findings useful and actionable for the user’s business or technical needs?

Specificity Check (The "Horoscope" Test):
Read the recommendations. Could they apply to ANY company without looking at this specific data? (e.g., "Improve customer satisfaction").
Penalty: If the advice is generic "consultant fluff" not tied to specific numbers found in the analysis, cap the score at 1. High scores require evidence-based specificity (e.g., "Target the under-25 demographic because their churn rose by 15%").

**Evaluation Guidance**:
- **0 (very poor)**: Fails to answer the main question. The report may include irrelevant information, generic statements, or be completely off-topic.
- **1 (poor)**: Touches on the topic but misses the core question. Answers are vague without specific details or actionable insights.
- **2 (fair)**: Answers the main question but misses secondary requirements or lacks specificity in details.
- **3 (good)**: Clearly answers the user's instruction. Covers all major requirements with specific findings, and offers insights or recommendations.
- **4 (excellent)**: Fully and precisely answers the instruction. The report is highly useful, tailored to the user's needs, and leaves no ambiguity.

---

### 2. Richness (Is the analysis deep and insightful?)

**Definition**: To what extent does the report provide non-trivial, multi-angled analysis?

**Key Evaluation Criteria**:
- **Depth of Insight**: Does the report go beyond simple counts/averages? Does it describe correlations, segments, trends, or causal factors?
- **Analytical Variety**: Does it mention and apply appropriate techniques (e.g., "We segmented users by region," "Regression analysis showed...")?
- **Non-Superficiality**: Is the analysis deeper than just surface-level aggregates?
- **Insight vs. Jargon**: Does the report explain why a complex technique (e.g., clustering, XGBoost) matters to the business question? Merely mentioning advanced models without interpreting the implication of the results should NOT guarantee a high score.

Information Density Check (The "Wikipedia" Trap):
Does the report explain definitions instead of data?
Penalty: If >30 percent of the text is educational filler (e.g., defining "Correlation" or "XGBoost") rather than interpreting the actual results, cap the score at 2. We want business insights, not a textbook lecture.

Appropriateness of Complexity (Occam's Razor): Does the report use the simplest effective method to answer the question?
Penalty: If the report applies complex machine learning models (e.g., Random Forest) to a problem that only required simple descriptive statistics (e.g., "What were total sales?"), penalize the score. High scores are for necessary depth, not performative complexity.

**Evaluation Guidance**:
- **0 (very poor)**: Extremely shallow analysis. The report acts merely as a raw data dump (e.g., listing rows) without summarizing patterns.
- **1 (poor)**: Basic aggregates only (counts, averages). No segmentation or trend analysis. Or, the report uses complex terminology incoherently ("word salad") without any logical connection to the data.
- **2 (fair)**: Moderate richness. The analysis identifies groups or trends, but remains descriptive (states *what* happened, but not *why*). **Crucially, if advanced methods are mentioned but not interpreted (e.g., "We used clustering" with no description of the clusters' business meaning), cap the score here.**
- **3 (good)**: Good depth and clarity. Findings are derived from multiple angles (e.g., segmentation, time series). **Technical results are translated into clear takeaways.** If a model/technique is used, the report explains what the result means for the user's question.
- **4 (excellent)**: Exceptionally rich and strategic. The report uses advanced perspectives (e.g., predictive factors, causality, anomalies) to uncover **non-obvious** business/clinical insights. **There is zero "empty jargon"**; every technical finding is fully integrated into a cohesive narrative that drives specific, high-value conclusions.

---

### 3. Soundness (Is the logic and narrative consistent?)

Definition: Is the reported story logically valid, internally consistent, and methodologically reasonable, based solely on the final report?

**Key Evaluation Criteria**:
- **Internal Consistency**: Are there contradictions in the findings or conclusions? Does the report maintain a logical flow from analysis to conclusion, without jumping to conclusions without evidence? Is there a clear connection between the reported analysis and the final conclusions?
- **Methodological Logic**: Are the analysis methods used appropriate and reasonable for the given problem? For example, are the right statistical or analytical techniques chosen based on the data type and problem at hand (e.g., using regression for numerical prediction, clustering for segmentation, or classification for categorical outcomes)? Is the choice of visualizations appropriate and relevant to the user's question? Are the columns and features used in the analysis well-chosen and aligned with the user's objectives?
- **Name-Dropping**: Does the report introduce advanced techniques, studies, or concepts without providing sufficient explanation or context? Are methods or techniques mentioned, but not explained in a way that shows their relevance to the current study? If methods are just "name-dropped" without sufficient discussion, it can weaken the logic and soundness of the report.
- **Narrative Integrity**: Does the text describe itself accurately? Check for "Meta-hallucinations". For example, if the intro claims "This is a 50-page deep dive" but the output is 3 paragraphs, or if it claims "As shown in the table below" but no table is present/described, this is a soundness failure.

Objectivity / Bias Check: Does the report blindly confirm the user's premise? If the analysis results contradict the user's implicit assumption (e.g., User asks "Why did sales drop?" but data shows sales are flat), the report MUST clarify this.
Bonus: Reward reports that correct a user's misconception based on data evidence.

**Evaluation Guidance**:
- **0 (very poor)**: Incoherent or internally contradictory. Methods and techniques are mentioned but not explained or applied appropriately. There is no clear connection between findings and conclusions, and name-dropping is prevalent without explanation or justification.
- **1 (poor)**: Weak logic. Jumps to conclusions without evidence or explanation. Techniques are used but not clearly explained or justified. Some methods may feel like "name-dropping" without being appropriately connected to the analysis.
- **2 (fair)**: Roughly coherent, but may have gaps in reasoning or weak justification for conclusions. Some methods are explained but may not be fully appropriate or adequately linked to the results. "Name-dropping" is minimal but still present without sufficient context.
- **3 (good)**: Clear and logical narrative. The methods are appropriately chosen and explained, though there may still be some room for improvement in depth. "Name-dropping" is minimal and explained adequately within the context of the analysis.
- **4 (excellent)**: Highly rigorous and consistent. Every method and technique is fully explained and justified in the context of the study. No "name-dropping"; each method is directly relevant and applied correctly, with clear connections to findings.
---

### 4. Visualization (Are the described/shown charts effective?)

**Definition**: How effectively does the report use visualizations to support understanding?

**Key Evaluation Criteria**:
- **Presence**: Does the report include or reference specific charts, graphs, or tables?
- **Relevance**: Do the descriptions of the charts support and enhance the narrative, or are they merely referenced with no explanation?
- **Interpretation**: Does the report explain what the chart shows and how it connects to the analysis?

Chart Logic Match: Is the described visualization type actually suitable for the data type described?
Bad Example: Using a Pie Chart for a time-series trend (12 months of data).
Bad Example: Using a Bar Chart to show a correlation between two continuous variables (should be Scatter).
Penalty: If the described chart type is functionally wrong for the data described in the text, cap the score at 2.

**Evaluation Guidance**:
- **0 (very poor)**: No visualizations mentioned or included when they are clearly needed to explain the data.
- **1 (poor)**: Visualizations are mentioned but are either irrelevant, too simplistic, or do not add value to the analysis.
- **2 (fair)**: Some useful visualizations are described, but there is limited variety or depth. Descriptions may lack clarity.
- **3 (good)**: A reasonable variety of well-chosen visualizations are included/described and directly support the analysis.
- **4 (excellent)**: Visualizations are integral to the report’s story. Descriptions are rich and insightful, providing clear and deep understanding of complex data patterns.

---

### 5. Readability (Is it a professional report?)

**Definition**: How well-written and well-organized is the final deliverable?

**Key Evaluation Criteria**:
- **Structure**: Are there clear headings (Intro, Findings, Conclusion)? Is the report logically organized and easy to follow?
- **Clarity**: Is the language professional, concise, and easy to understand? Is the report free of jargon or overly complex language?
- **Formatting**: Is the report well-formatted (bullet points, headings, text alignment)?

**Evaluation Guidance**:
- **0 (very poor)**: Disorganized, messy, or extremely hard to read. The report may lack clear structure or have numerous grammatical errors.
- **1 (poor)**: Rough draft quality. Poor formatting or unclear writing. Difficult to follow or scan.
- **2 (fair)**: Acceptable structure but lacks clarity or may be too dense. Some sections may be difficult to understand.
- **3 (good)**: Professional tone. Good use of formatting (headings, bullet points). Easy to scan and follow.
- **4 (excellent)**: Polished, executive-level quality. Excellent structure, clear executive summary, professional formatting, and precise language.

Provisional score for Readability: X/4

---

### Evaluation Procedure

Follow this procedure **step by step**.

1. **Read the User Instruction and Report thoroughly.**
2. **Safety Check: Language Integrity**:
  - Immediately scan for gibberish, random tokens, or chaotic language mixing.
  - If found: STOP the detailed analysis. Explicitly state the failure in your critique and assign **0 for ALL dimensions** in the final JSON.
3. **Dimension-by-dimension analysis** :
   For each dimension:
   - Quote a specific sentence or section from the report that justifies your score.
   - Critique: What is missing? What is good?
   - Score: Assign an integer (0-4).
   
   *Critical Check*: When evaluating Task Fulfillment, Soundness, and Visualization, explicitly check if the report answers the relevant questions in the Instruction. For example:
   - Does the report directly address the relevant question(s)? If the user asks "Why did sales drop?", the report should provide an explanation, not just show the sales percentage change.
   - Are the analysis methods appropriate and relevant to the user’s question? The methods should be logically chosen based on the data type and the problem posed. For example, if the user is asking about trends over time, time series analysis or similar methods should be used; if segmentation is required, the report should segment the data accordingly.
   - Are the data columns used relevant to the user's question? Ensure that the report uses the appropriate features for the analysis. For instance, if the question is about customer behavior, the data should include columns like customer ID, purchase history, etc., and avoid irrelevant columns that don’t contribute to the answer.
   - Are the visualizations meaningful and relevant to the user's question? The report should include visualizations that directly support the narrative and analysis. For example, if the user wants to understand how two variables are related, a scatter plot or correlation matrix would be appropriate. Avoid using visualizations that are forced or unrelated to the data at hand.

4. **Final JSON output**  
   After Writing down a thorough and detailed evaluation, output one JSON code block. Ensure that the reasoning behind each score is clearly detailed in the earlier steps.

```json
{{
  "task_fulfillment": <integer from 0 to 4>,
  "richness": <integer from 0 to 4>,
  "soundness": <integer from 0 to 4>,
  "visualization": <integer from 0 to 4>,
  "readability": <integer from 0 to 4>
}}
```
"""

REPORT_SOUNDNESS_EVALUATE_PROMPT = """
You are an Expert Data Science Auditor.

You are given:
- a user instruction describing a data analysis task, and
- a **full execution trajectory plus final report** produced by a data analysis agent.

### Structure of [MODEL RESPONSE]

In the [MODEL RESPONSE], the components follow this structure:

1. **Execution Trajectory** (multiple steps), including:  
   - `<Analyze>...</Analyze>`: the agent’s internal analysis / planning thoughts.
   - `<Understand>...</Understand>`: the agent’s interpretation of the instruction or data.
   - `<Code>...</Code>`: Python code that the agent proposes to execute.
   - `<Execute>...</Execute>`: code execution results and logs (the actual outputs of running the code).

2. **Final Report**:
   - The final user-facing report is wrapped in `<Answer>...</Answer>` at the **end** of the [MODEL RESPONSE].

You must treat the content inside `<Execute>...</Execute>` as the **only factual source of truth** about what actually happened with the data (numbers, models, tests, plots, etc.).

- `<Code>` shows what the agent tried or intended to run, but only `<Execute>` confirms what was actually executed and what results were obtained.
- `<Analyze>` and `<Understand>` contain reasoning and interpretation, but not ground-truth data.
- The Final Report (`<Answer>...</Answer>`) must be consistent with the Execution Trajectory, and the overall process should revolve around answering the main task in the [INSTRUCTION].

Your goal is to evaluate the **soundness of the entire process** and the consistency between trajectory and report.

You will rate the solution on the following 5 dimensions, each from 0 (lowest) to 4 (highest):

1. **Faithfulness (Report–Trajectory Consistency)**  
2. **Process–Instruction Alignment (Does the analysis process stay focused on the task?)**  
3. **Analytical Depth (Richness and sophistication of relevant techniques)**  
4. **Process Coherence (Does the sequence of steps form a clear, logical pipeline?)**  
5. **Visualization & Evidence (Use, relevance, and richness of visual support)**  

---

### DIMENSION DEFINITIONS

#### 1. Faithfulness (CRITICAL – The "Truth" Check)

**Core question**:  
Does the Final Report (`<Answer>`) faithfully reflect what actually happened in the Execution Trajectory (`<Code>` + `<Execute>`)?

You must systematically check that:

- Every **important number** in the Final Report (metrics, counts, percentages, p-values, coefficients, etc.) appears in or is directly derivable from some `<Execute>` output.
- Every **method / model / test** claimed in the Final Report (e.g., “Random Forest”, “t-test”, “linear regression”, “k-means”) was actually executed in the trajectory (code + successful execution logs).
- Every **described visual pattern or figure** in the Final Report (e.g., “strong upward trend in time series”, “two clear clusters”, “right-skewed distribution”) corresponds to a plot or numeric summary produced in `<Execute>`.

Hallucinations and inconsistencies include (non-exhaustive):
- Numbers or metrics in the report that **never appear** in the logs.
- Methods, tests, or models claimed in the report but **never actually executed**.
- Descriptions of plots or patterns that **do not match** any generated outputs.
- Narrative statements that **contradict** what logs show (e.g., report says “no missing values” but logs show many NULLs).

Traceability Requirement: It is NOT enough for a number to merely "appear" in the logs. You must verify derivation:
Does the <Code> actually contain the logic/formula to calculate that number?
Did the Agent just hardcode a number into a print statement (e.g., print("50%")) without calculating it from data?
Hardcoded or hallucinated derivations receive a score of 0.

The "Silent Failure" Check: Look closely at the <Execute> logs.
Did the code output Empty DataFrame, [], NaN, or suppress an error?
If the Agent proceeds to draw conclusions or "Answer" the user request based on empty or failed execution outputs, this is a critical failure.
Score: Faithfulness must be 0 in this case, regardless of how polite the text is.

Dependence Check: The number must be derived from operations on the dataset (e.g., df['col'].mean(), model.predict()), NOT merely assigned from a constant or a user input variable.
Penalty: If a metric is just a variable assignment (e.g., x = 10; print(x)) without any data computation, treat it as Unsupported/Hallucinated (Score 0).

Contradictory Interpretation: The numbers match, but the conclusion drawn from them is scientifically wrong. (e.g., The log shows p-value=0.8 or correlation=0.01, but the report claims "significant difference" or "strong relationship").
Penalty: If the Agent accurately cites a number but flips its meaning to sound positive, Score = 0.

Logic-Output Verification: Check the <Code> against the <Execute> output.
Did the agent print a conclusion (e.g., "No Nulls") that is hardcoded in a print string ignoring the actual calculation variable?
Example Failure: count = df.isna().sum(); print("No issues") -> The print statement does not depend on count.
Requirement: The output/conclusion must be dynamically generated from the variable. Hardcoded confirmation messages get Score 0.

Variable Flow Verification (The "Shell Game" Hack): You must Trace the variable printed in <Execute> back to its origin in <Code>.
Crucial: Does the printed variable actually hold the result of the analysis function?
Trap: Watch out for agents that calculate a metric into variable A, but then print variable B (which is hardcoded or unrelated).
Penalty: If the data flow is broken or swapped, Score 0.

**Guideline scores:**
- **0 (Fail)**: Clear hallucinations or contradictions. Multiple important claims in the report are unsupported or contradicted by the trajectory.
- **1 (Low)**: Several suspicious or weakly supported claims; some numbers or methods cannot be reliably linked to logs.
- **2 (Passable)**: Mostly consistent; some claims are vague or only loosely connected to logs, but no major contradictions or obvious hallucinations.
- **3 (Good)**: Strong consistency. Most important claims can be traced to specific execution results. Very little speculative content.
- **4 (Perfect)**: Completely faithful. All important claims, numbers, and described methods/plots are clearly supported by execution outputs. No hallucinated metrics, methods, or visuals.

---

#### 2. Process–Instruction Alignment (Does the analysis process stay focused on the user’s task?)

**Core question**:  
Does the **analysis process itself** (the sequence of `<Analyze>/<Understand>/<Code>/<Execute>` steps) clearly focus on answering the user’s instruction, using techniques and visualizations that are relevant and helpful for solving that task?

This dimension is about the **trajectory**, not re-evaluating the final report alone (the report quality is evaluated separately).

You should check:
- Whether the main steps of the analysis are **designed around the instruction** (e.g., if the user asks for churn drivers, the steps should focus on churn-related features, models, and metrics).
- For each major technique or plot used, whether it is:
  - **Relevant and helpful** for answering the instruction, or
  - **Irrelevant / distracting**, adding confusion and not helping the user’s question.

Data Scope Integrity Check: Did the agent arbitrarily filter or drop data without a valid statistical reason (e.g., outlier removal), just to make the analysis easier or the results look better?
Example: User asks for "yearly performance", Agent filters to "Q4 only" without explanation.
Penalty: This is a manipulation of the task. Score 0.

Advanced or fancy techniques that are **irrelevant to the user’s task** should **not** increase this score; they are a negative signal.

**Guideline scores:**
- **0 (Fail)**: The process largely ignores the instruction. Many steps analyze unrelated aspects or datasets; many techniques and plots have no clear relation to the user’s question.
- **1 (Weak)**: Some steps clearly address the instruction, but a large portion of the trajectory is off-topic or only loosely related. Several methods or plots are unnecessary for the task.
- **2 (Partial)**: The process is partly aligned. The main direction of the analysis relates to the instruction, but there are noticeable detours or segments that do not clearly help answer the user’s question.
- **3 (Good)**: Most steps are clearly motivated by the instruction. Techniques and plots are chosen because they help address the task, with only minor unnecessary digressions.
- **4 (Excellent)**: The entire process is tightly focused on the instruction. Almost every major step, technique, and plot is clearly relevant and helpful for answering the user’s question, with virtually no wasted or off-topic analysis.

---

#### 3. Analytical Depth (Richness and sophistication of **relevant** techniques)

**Core questions:**
- Among the **relevant and helpful** parts of the analysis, how deep and sophisticated are the techniques used?
- Does the analysis go beyond trivial inspection (basic EDA) to use more advanced and layered methods where appropriate?
- Conditional on relevance: does using richer, more advanced techniques and multiple meaningful angles increase the depth?

Examples of techniques:
- Basic: `.head()`, `.describe()`, simple counts, single aggregates.
- Intermediate: groupby, segmentation by categories/time, correlation matrices, distributions, outlier checks.
- Advanced: statistical tests (t-tests, ANOVA, chi-square), regression/classification models, clustering, time-series models, dimensionality reduction, feature engineering, complex pipelines.

Important principles:
- **Relevance is a prerequisite**: clearly irrelevant or unhelpful techniques should **not** be counted as depth, and may indicate a noisy process.
- Among the **relevant** techniques:
  - Using only very basic EDA, even if relevant, should lead to at most a moderate score (around 3).
  - Using appropriate advanced methods (e.g., models, statistical tests, feature engineering) that clearly help answer the question should raise the score.
  - Using a richer combination of relevant techniques and perspectives (e.g., descriptive + correlation + modeling + diagnostics) should yield higher depth scores.
- **Execution vs. Interpretation**: Merely importing a library or running a function (e.g., model.fit()) counts as Low Depth if the agent does not inspect or interpret the outputs. **True Depth** requires evidence of **using** the results: extracting feature importance, analyzing cluster centroids, checking residuals, or summarizing statistical test p-values. **Penalty**: If the agent runs an advanced method but fails to print/analyze the specific outcomes (e.g., runs clustering but doesn't check what the clusters represent), score this as **Superficial** (max score 2), regardless of how "advanced" the algorithm is.

Critical Penalty for Over-Engineering: If the agent applies complex models (e.g., Machine Learning, Deep Learning) to a problem that could be solved with simple aggregation or statistics, AND fails to justify why the complex model is needed, this is Over-Engineering.
Score Penalty: Max score is 2.
Example: Running a Random Forest just to calculate the average sales per region.
True Depth requires the method to be the right tool for the complexity of the problem.

Scientific Validity Check: Complexity without correctness is meaningless. You must penalize methodological flaws, even if the code runs without errors.
Data Leakage: If the agent trains and evaluates a model on the same dataset (no train/test split) to claim high performance, this is a critical flaw. Max Depth Score: 1.
Invalid Assumptions: Applying tests (e.g., T-test) without checking assumptions, or treating categorical IDs as numerical values for correlation, is Flawed Analysis. Max Depth Score: 2.
True Depth implies correct application of statistical principles.

The "Lazy Conclusion" Penalty: If the agent performs only basic checks (counts, missing values) and then concludes "No issues found" or "Analysis complete" without attempting to dig deeper (e.g., exploring segments, correlations, or time trends) when the instruction clearly implies a need for insight:
This is Superficial Analysis.
Max Score: 1, even if the basic checks are correct. Depth requires effort to uncover insights, not just checking for data errors.

Metric Mismatch: Using the wrong metric for the data context (e.g., reporting only 'Accuracy' for a highly imbalanced dataset like fraud detection, without checking Recall/F1/AUC).
Penalty: This is a fundamental data science error. Max Depth Score: 1.

In other words:
- Techniques that are **relevant + advanced + actually used in reasoning and conclusions** should significantly increase the depth score.
- Techniques that are advanced but **irrelevant or unused** should not increase the depth score and may harm Process–Instruction Alignment or Process Coherence instead.

**Guideline scores:**
- **0 (Very shallow)**: The relevant part of the analysis consists almost entirely of trivial operations (e.g., `.head()`, `.describe()`, a few simple counts) with no real depth or variety.
- **1 (Weak)**: Some non-trivial relevant analysis (e.g., groupby, simple segmentations), but mostly basic; almost no use of more advanced relevant techniques.
- **2 (Standard)**: A reasonable set of relevant techniques: basic EDA plus some intermediate analyses (e.g., correlations, multi-level groupby, distributions, simple tests/models). Depth is acceptable but not especially rich.
- **3 (Deep)**: Good depth with multiple relevant techniques: e.g., solid EDA, segmentation, correlations, and at least one or two suitable advanced methods (tests, models, feature engineering) that clearly contribute to answering the question.
- **4 (Very deep)**: High depth with a rich, well-chosen set of relevant techniques: multiple complementary advanced methods (e.g., tests + models + diagnostics + thoughtful feature engineering), all clearly linked to the question and used to extract nuanced insights.

---

#### 4. Process Coherence (Does the sequence of steps form a clear, logical pipeline?)

**Core questions:**
- Does the analysis follow a **natural, understandable pipeline** (e.g., data understanding → exploration → modeling/testing → evaluation → conclusion)?
- Are the steps connected in a logical way, or does the agent jump randomly between unrelated methods?
- Are advanced techniques integrated into the flow, or just dropped in as isolated experiments?
- Does the execution demonstrate smoothness and robustness, or is the flow fragmented by frequent restarts due to avoidable low-level errors (e.g., SyntaxErrors, NameErrors, missing dependencies)?

You should focus on the **global structure** of the trajectory:

- Whether each major step prepares or motivates the next step.
- Whether there is a clear main storyline that a human analyst could follow.
- Whether there is excessive “method dumping”: trying many unrelated methods without integrating them into a coherent narrative.

Efficiency Check:
If a significant portion (>30%) of the Execution Trajectory consists of failed code blocks followed by corrections (trial-and-error), strictly limit the score to max 2, even if the final result is correct.
A coherent process implies competence, not just persistence.

Data Hygiene Requirement: A coherent pipeline MUST start with basic checks.
Did the agent check for missing values, duplicates, or data types before running complex models?
If the agent feeds "dirty data" (e.g., columns with NaNs or wrong types) directly into a model/calculation without any inspection or cleaning steps, penalize Process Coherence.
Score Penalty: A pipeline that ignores obvious data quality issues cannot score above 2, as it is logically fragile.

**Guideline scores:**
- **0 (Chaotic)**: The trajectory feels disorganized. The agent jumps between unrelated operations and methods with no clear purpose or pipeline. It is hard to reconstruct a meaningful story.
- **1 (Weak)**: Some local structure exists, but the overall flow is poor. There are many detours, restarts, or unrelated experiments. The main storyline is hard to follow.
- **2 (Acceptable)**: A loose pipeline exists (e.g., basic EDA → some focused analyses → some form of modeling or conclusion), but there are redundancies, poorly connected steps, or the execution is clumsy (e.g., relies on a 'fix-it-until-it-works' approach with multiple avoidable crashes like SyntaxErrors). Coherence is mediocre but not terrible.
- **3 (Coherent)**: A clear, mostly linear or well-structured pipeline is present. Steps follow naturally; advanced methods are introduced at sensible points and integrated into the overall reasoning.
- **4 (Highly coherent)**: The trajectory shows an excellent, well-organized pipeline with robust execution (no avoidable crashes). Each major step is motivated by previous observations and leads naturally to the next. There is almost no random method dumping; the whole process reads like a carefully planned analysis.

---

#### 5. Visualization & Evidence (Use, relevance, and richness of visual support)

**Core questions:**
- Does the trajectory make meaningful use of visualizations and other concrete evidence?
- Are the visualizations **relevant** to the user’s task and the main analytical questions?
- Given relevance as a prerequisite, how **rich and diverse** is the visual support?

You must:
- Prioritize **relevance and correctness** first.
- Only then reward **richness / variety**: more relevant, well-explained plots should lead to higher scores.

If the agent generates "filler charts" that are technically correct but provide no insight to answer the user question, cap the score at 2. Quality Check: Titles and Labels must be Descriptive.
Generic labels like title="Plot", xlabel="x", ylabel="y" should be treated the same as missing labels (Max score 2).
High scores require business-relevant labels (e.g., title="Monthly Sales Trend 2023", ylabel="Revenue ($)").

The "Blind Judge" Rule (Code Inspection): Since you cannot see the actual images, you must judge the code quality of the plots.
Does the plotting code explicitly set a title, xlabel, and ylabel?
Does it add a legend if there are multiple lines/groups?
Penalty: plt.plot(df['col']) without any labels/titles is Basic (max score 2), even if the data is relevant. To get a 3 or 4, the code must demonstrate an intent to make the plot self-explanatory (labels, titles, proper formatting).

Legibility Heuristic: Check the complexity of the plot code.
Does the agent try to plot too many categories or variables at once (e.g., a bar chart with 50 bars, a heatmap with 100 features, or a pairplot of the entire dataset)?
Penalty: If the plot is theoretically "unreadable" due to overcrowding, it adds zero value. Cap score at 2. High-quality visuals focus on top-k items or specific aggregations, not dumping all data onto a canvas.

**Guideline scores:**
- **0 (None / Misleading)**: No plots generated when they would clearly help; or the report describes visual patterns that are not backed by any executed plots. Visual evidence is absent or misleading.

Deceptive Scaling Check: Check the plotting code for misleading manipulations.
Truncated Y-Axis: Does the agent set plt.ylim() to exaggerate small differences (e.g., zooming in on 99-100 to make a difference look huge)?
Penalty: If the visualization distorts the data reality to support a weak narrative, Score 1.

- **1 (Weak)**: One or two basic plots, but they add limited value, are poorly labeled, or only weakly connected to the task.
- **2 (Basic)**: Some helpful and relevant plots that illustrate key points, but variety or depth is limited. Visuals are somewhat useful but not rich.
- **3 (Good)**: Several well-chosen, relevant plots that clearly support the analysis and are connected to the task. Some diversity of chart types and clear interpretation.
- **4 (Rich & Meaningful)**: A rich set of relevant, well-explained plots (possibly multiple types) that strongly illuminate the data and conclusions. All visuals are clearly tied to the task and help the user understand key patterns.

If many plots are generated but are unrelated to the instruction or never used in reasoning, this should **lower** the score.

---

### [INSTRUCTION]:
{instruction}

### [MODEL RESPONSE] (Execution Trajectory + <Answer> Final Report):
{model_response}

---

### EVALUATION PROCEDURE AND OUTPUT REQUIREMENTS

Follow this procedure **step by step**.  
Do **not** skip any step or dimension.

#### Step 1: Reconstruct the full analysis pipeline (MANDATORY)

1. Carefully read the Execution Trajectory (all `<Analyze>`, `<Understand>`, `<Code>`, `<Execute>` blocks).
2. Reconstruct the **main pipeline** of the analysis as a sequence of steps in **chronological order**.
   - For each meaningful step, write a line such as:  
     `Step k: <brief description of what was done and why>`,  
     for example: “Step 3: Loaded dataset X and inspected columns”,  
     “Step 5: Computed summary statistics grouped by customer segment”,  
     “Step 7: Fitted a logistic regression model for churn prediction”, etc.
   - Include **all significant steps** that involve data operations, analysis, modeling, testing, plotting, or key reasoning.  
     Do **not** skip steps; be exhaustive at a reasonable level of granularity.
3. This pipeline reconstruction will be your reference for all subsequent evaluations.

#### Step 2: Dimension-by-dimension audit (MANDATORY)

For each dimension in the following order:

1. **Faithfulness**  
2. **Process–Instruction Alignment**  
3. **Analytical Depth**  
4. **Process Coherence**  
5. **Visualization & Evidence**

You MUST:

1. Write a short heading, for example:  
   `Dimension 1 – Faithfulness`

2. Follow these **dimension-specific analysis requirements**:

---

**(A) Faithfulness**

- Extract the key claims from the Final Report `<Answer>`, including:
  - Important numerical values (metrics, counts, percentages, p-values, coefficients, etc.).
  - Named models and methods (e.g., “Random Forest”, “logistic regression”, “ANOVA test”).
  - Described visual patterns or figure-based statements (e.g., “clear upward trend”, “two clusters”, “right-skewed distribution”).
  - Major conclusions that rely on quantitative or visual evidence.
- For **each** key claim, explicitly indicate:
  - Whether you can find a corresponding **supporting step** in the Execution Trajectory (a specific `<Execute>` output, or a clearly executed `<Code>` + `<Execute>` pair).
  - Or whether you **cannot** find any supporting evidence (treat this as hallucinated or unsupported).
- Clearly list:
  - **Supported claims** (with brief references to which steps support them), and
  - **Unsupported / hallucinated claims**.
- Based on this, explain how faithful the Final Report is to the trajectory.

End this section with:  
`Provisional score for Faithfulness: X/4`

---

**(B) Process–Instruction Alignment**

- Briefly restate, in your own words, what the user is asking in the [INSTRUCTION].
- Using your reconstructed pipeline from Step 1:
  - For each major step, indicate whether it is:
    - **Directly relevant** to the user’s question,
    - **Indirectly useful** (e.g., generic data checks that are still reasonable), or
    - **Irrelevant / distracting** (not helping the task).
- Explicitly list:
  - The main **relevant techniques and steps** that clearly help address the instruction.
  - The main **irrelevant or unnecessary techniques/steps** that do not help answer the question.
- Explain whether the majority of the process is centered on the task or not.

End this section with:  
`Provisional score for Process–Instruction Alignment: X/4`

---

**(C) Analytical Depth**

- From the pipeline, list all **distinct analytical techniques** actually executed (based on `<Code>` + `<Execute>`), for example:
  - basic EDA (head, describe, simple summaries),
  - groupby / segmentation,
  - correlations / covariance,
  - distribution / outlier analysis,
  - statistical tests,
  - regression / classification models,
  - clustering,
  - time-series models,
  - feature engineering, etc.
- For each technique, indicate:
  - Whether it is **relevant** to the instruction (helps answer the user’s question), or mostly irrelevant.
  - Whether it is **basic / intermediate / advanced**.
- Clearly separate:
  - **Relevant techniques that deepen understanding** and contribute to the final reasoning, and
  - Techniques that are advanced but **irrelevant or unused** in the reasoning.
- Based primarily on the **relevant** techniques:
  - Comment on the depth: is it very shallow, moderate, deep, or very deep?
  - Explain why (e.g., “only basic EDA”, “EDA + correlations + regression model with proper evaluation”, etc.).

End this section with:  
`Provisional score for Analytical Depth: X/4`

---

**(D) Process Coherence**

- Using your pipeline reconstruction, describe the **overall flow** of the analysis in your own words:
  - Does it follow a natural sequence (e.g., data understanding → exploration → modeling/testing → evaluation → conclusion)?
  - Or does it frequently jump between unrelated methods (e.g., randomly switching between regression, clustering, correlations) with no clear storyline?
- For each major step, briefly check:
  - What its purpose is in the context of the pipeline.
  - Whether it naturally follows from previous steps and prepares the next ones.
- Explicitly identify:
  - Steps or method blocks that are clearly integrated into a coherent pipeline, and
  - Steps or method blocks that look like scattered, unjustified experiments.
  - Disruptive technical failures: Instances where the pipeline broke due to avoidable coding errors (e.g., SyntaxErrors, NameErrors, missing dependencies) forcing unnecessary restarts.
- Explain clearly whether the analysis feels like:
  - a coherent pipeline with meaningful flow and robust execution (without frequent crashes due to basic mistakes), or
  - a stumbling process where the logic is sound but the execution is clumsy (e.g., "fix-it-until-it-works"), or
  - a scattered collection of experiments and method-dumping.

End this section with:  
`Provisional score for Process Coherence: X/4`

---

**(E) Visualization & Evidence**

- From the Execution Trajectory:
  - List all major plotting / visualization operations that appear in `<Code>` and `<Execute>` (e.g., bar charts, line charts, scatter plots, histograms, box plots, heatmaps, time-series plots).
- For each major visualization type:
  - State whether it is **relevant and helpful** for the user’s question and the main analysis, or
  - Mostly **decorative / redundant / unrelated**.
- From the Final Report `<Answer>`:
  - Identify any described visual patterns or referenced figures.
  - Check whether each such description can be traced back to an executed plot or numeric summary.
- Explicitly list:
  - **Relevant and helpful visuals**, and
  - **Irrelevant, unused, or hallucinated visuals**.
- Based on this, explain:
  - Whether the solution uses visual evidence meaningfully, and
  - Given relevance as a prerequisite, whether the visual support is **rich** (diverse and informative) or limited.

End this section with:  
`Provisional score for Visualization & Evidence: X/4`

Before finalizing your scores, ask yourself these "Anti-Gaming" questions. If the answer is YES, downgrade the relevant score immediately:
1. **The "Tautology" Trap**: Did the agent just print a number it defined itself (e.g., `x=5; print(x)`) without calculating it from the data? -> Faithfulness = 0.
2. **The "Log Dump" Trap**: Did the agent print thousands of lines of raw logs (e.g., `print(df.to_string())`) to hide the fact that it didn't analyze anything? -> Coherence = max 1.
3. **The "Unreadable Plot" Trap**: Did the agent plot 20+ variables or 50+ categories in a single chart (overcrowding), making it likely unreadable? -> Visualization = max 2.
4. **The "Lazy Conclusion" Trap**: Did the agent run only basic `.info()` checks and declare the job done, avoiding the hard analytical work? -> Analytical Depth = max 1.
5. **The "Silent Failure" Trap**: Did the logs contain errors or empty dataframes that agent ignored? -> Faithfulness = 0.

---

#### Step 3: JSON Output (MANDATORY, at the very end)

After you have completed:

- the pipeline reconstruction, and
- the detailed analysis and provisional score for **all five dimensions**,

you must output **exactly one** JSON code block in the following format:

```json
{{
  "faithfulness": <integer score 0-4>,
  "instruction_answering": <integer score 0-4>,
  "analytical_depth": <integer score 0-4>,
  "process_coherence": <integer score 0-4>,
  "visualization": <integer score 0-4>,
  "reason": "<Brief one- or two-sentence summary of the overall assessment>"
}}
```

Where:

"faithfulness" = your final score for the Faithfulness dimension.

"instruction_answering" = your final score for Process–Instruction Alignment.

"analytical_depth" = your final score for Analytical Depth.

"process_coherence" = your final score for Process Coherence.

"visualization" = your final score for Visualization & Evidence.

Requirements:

Each score must be an integer from 0 to 4.

The scores in the JSON must match the provisional scores you stated for each dimension.

"reason" should briefly summarize the main strengths and weaknesses of the trajectory and the report–trajectory consistency.

The JSON must be valid (no comments, no trailing commas).

Do not include any other JSON objects outside this code block.

Do not write any text after the JSON code block.
"""