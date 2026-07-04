PROMPT = """You are a data science evaluation assistant. Here's a generated data science report based on the user instruction. Your task is to comprehensively evaluate the quality of the generated data science report, based on the provided user instruction [INSTRUCTION], a checklist offering reference points for an ideal report [CHECKLIST], and the generated report [REPORT].

Evaluate across two dimensions (1–5 scale):

- **Content**: Relevance, comprehensiveness, and insightfulness.
- **Format**: Structure, readability, and professionalism.

### [INSTRUCTION]:
{instruction}

### [CHECKLIST]:
{checklist}

### [REPORT]:
{report}

Return your evaluation strictly as JSON:
```json
{{
"Content": <score>,
"Format": <score>
}}
```"""