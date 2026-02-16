"""
Prompt templates for FinQA Program-of-Thought system
"""

FINQA_PROGRAM_GENERATION_PROMPT = """You are a financial reasoning expert. Given a financial question and retrieved evidence, generate a program to compute the answer.

RULES:
1. Use ONLY these operations:
   - add(a, b): addition
   - subtract(a, b): subtraction  
   - multiply(a, b): multiplication
   - divide(a, b): division
   - percent(a, b): compute a as percentage of b
   - diff(a, b): absolute difference
   - max(list): maximum value
   - min(list): minimum value
   - sum(list): sum of values
   - avg(list): average of values

2. Use ONLY numbers that appear in the evidence below
3. Each line should be: variable = operation(args)
4. Return final answer in the last variable

RETRIEVED EVIDENCE:
{evidence}

QUESTION: {question}

Generate the program step by step. First write your reasoning, then output the program.

FORMAT:
REASONING:
[Your step-by-step reasoning]

PROGRAM:
const_0 = [first_number]
const_1 = [second_number]
result_0 = operation(const_0, const_1)
...
answer = [final_variable]

Now generate:
"""


FINQA_PROGRAM_REPAIR_PROMPT = """The previous program failed to execute or contained numbers not in evidence.

ERROR: {error}

ORIGINAL PROGRAM:
{program}

EVIDENCE:
{evidence}

QUESTION: {question}

Generate a corrected program that:
1. Uses ONLY numbers from the evidence
2. Uses valid operations
3. Has correct syntax

CORRECTED PROGRAM:
"""


EDGAR_ANSWER_GENERATION_PROMPT = """You are a financial analyst answering questions about SEC filings.

RETRIEVED EVIDENCE:
{evidence}

QUESTION: {query}

INSTRUCTIONS:
1. Answer based ONLY on the evidence provided
2. Include "As of [DATE]" with the filing date
3. Cite sources using [doc_id:chunk_id] format
4. If you reference specific numbers, cite them
5. Be precise and professional
6. If information is not in evidence, say "Information not found in available filings"

ANSWER:
"""


EDGAR_ANSWER_WITH_TABLES_PROMPT = """You are a financial analyst answering questions about SEC filings with table data.

RETRIEVED TEXT EVIDENCE:
{text_evidence}

RETRIEVED TABLES:
{table_evidence}

QUESTION: {query}

INSTRUCTIONS:
1. Analyze both text and table data
2. Extract relevant numbers from tables
3. Include "As of [DATE]" with the filing date
4. Cite sources using [doc_id:chunk_id] format
5. For table data, mention the table and row/column
6. Be precise with numerical values
7. If information is not in evidence, state this clearly

ANSWER:
"""


NUMERIC_VERIFICATION_PROMPT = """Verify if the numeric claims in the answer match the evidence.

ANSWER: {answer}

EVIDENCE: {evidence}

Extract all numeric claims from the answer and check if they appear in the evidence.

OUTPUT FORMAT:
VERIFIED NUMBERS:
- [number]: [true/false] - [source location or "not found"]

UNVERIFIED CLAIMS:
- [list any numbers that don't appear in evidence]

VERIFICATION RESULT: [PASS/FAIL]
"""


TABLE_SUMMARY_PROMPT = """Summarize this financial table in 2-3 sentences. Focus on:
1. What the table shows (metrics, time period, entities)
2. Key numbers or trends
3. Context (e.g., "revenue by quarter", "balance sheet items")

TABLE:
{table}

SUMMARY:
"""


PROGRAM_DEBUG_PROMPT = """Debug this financial calculation program.

PROGRAM:
{program}

ERROR:
{error}

AVAILABLE NUMBERS IN EVIDENCE:
{available_numbers}

Identify the issue and suggest a fix. Common issues:
- Using numbers not in evidence
- Wrong operation order
- Syntax errors
- Division by zero

DIAGNOSIS:
"""


EVIDENCE_RELEVANCE_PROMPT = """Rate how relevant this evidence is for answering the question.

QUESTION: {question}

EVIDENCE:
{evidence}

Does this evidence contain information needed to answer the question?
Consider:
- Does it contain relevant numbers?
- Does it provide necessary context?
- Is it about the right topic/entity?

RELEVANCE SCORE (0-10): 
EXPLANATION:
"""


XBRL_VERIFICATION_PROMPT = """Verify numeric values against XBRL data.

CLAIMED VALUES: {claimed_values}

XBRL DATA:
{xbrl_data}

For each claimed value:
1. Find matching XBRL tag
2. Compare values (allow ±0.1% tolerance for rounding)
3. Note any discrepancies

VERIFICATION:
"""


def get_program_generation_prompt(question: str, evidence: str) -> str:
    """Get prompt for FinQA program generation"""
    return FINQA_PROGRAM_GENERATION_PROMPT.format(
        question=question,
        evidence=evidence
    )


def get_program_repair_prompt(question: str, evidence: str, program: str, error: str) -> str:
    """Get prompt for program repair"""
    return FINQA_PROGRAM_REPAIR_PROMPT.format(
        question=question,
        evidence=evidence,
        program=program,
        error=error
    )


def get_edgar_answer_prompt(query: str, evidence: str) -> str:
    """Get prompt for EDGAR answer generation"""
    return EDGAR_ANSWER_GENERATION_PROMPT.format(
        query=query,
        evidence=evidence
    )


def get_edgar_answer_with_tables_prompt(query: str, text_evidence: str, table_evidence: str) -> str:
    """Get prompt for EDGAR answer with tables"""
    return EDGAR_ANSWER_WITH_TABLES_PROMPT.format(
        query=query,
        text_evidence=text_evidence,
        table_evidence=table_evidence
    )


def get_numeric_verification_prompt(answer: str, evidence: str) -> str:
    """Get prompt for numeric verification"""
    return NUMERIC_VERIFICATION_PROMPT.format(
        answer=answer,
        evidence=evidence
    )


def get_table_summary_prompt(table: str) -> str:
    """Get prompt for table summarization"""
    return TABLE_SUMMARY_PROMPT.format(table=table)


def get_program_debug_prompt(program: str, error: str, available_numbers: str) -> str:
    """Get prompt for program debugging"""
    return PROGRAM_DEBUG_PROMPT.format(
        program=program,
        error=error,
        available_numbers=available_numbers
    )
