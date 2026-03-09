You are an expert SQL developer and database instructor.

Given the following SQL table schema:

{ddl}

Generate exactly {n_samples} diverse question-SQL training pairs for the table above.

Rules:
- Cover a wide variety of SQL patterns: SELECT *, WHERE filters, ORDER BY, LIMIT, GROUP BY, aggregate functions (COUNT, SUM, AVG, MIN, MAX), HAVING, subqueries, and CASE expressions.
- Vary the natural language phrasing — avoid repeating the same sentence structure.
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
