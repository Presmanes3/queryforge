You are an expert SQL developer and database instructor.

Given the following SQL table schema:

{ddl}

Generate exactly {n_samples} diverse question-SQL training pairs for the table above.

This is batch {batch_index}. Focus primarily on these SQL patterns: {pattern_focus}.

Rules:
- Prioritise the SQL patterns listed above for this batch.
- Vary the natural language phrasing — avoid repeating the same sentence structure.
- Do not reuse questions that are obvious paraphrases of each other.
- Every SQL query must be syntactically valid for SQLite.
- Do not use JOINs with tables that are not defined in the schema above.
- Each question must be answerable by a single SQL query against the table.

Respond with a JSON array and nothing else. Each element must have exactly two keys:
- "question": the natural language question (string)
- "sql": the SQL query that answers it (string)

Example format:
[
  {{"question": "How many orders are there?", "sql": "SELECT COUNT(*) FROM orders;"}},
  {{"question": "What is the total amount across all orders?", "sql": "SELECT SUM(amount) FROM orders;"}}
]
