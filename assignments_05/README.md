# Week 5: Introduction to AI

## Setup

From the repository root:

```console
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` at the repository root and replace the placeholder
with the course-provided OpenAI API key. Never commit `.env`.

```console
python assignments_05/warmup_05.py
python assignments_05/project_05.py --demo
python assignments_05/project_05.py
```

The warmup runs every required API, persona, and prompt exercise with labeled
outputs. The project demo runs the starter bullet rewrite, cover letter test,
and passing/flagged moderation examples. The default project command starts the
interactive chatbot. Type `quit` or `exit` at any prompt to leave.

## Before submission

- Run the scripts with the course key; actual API outputs were not available in
  the authoring environment. Comments distinguish expected behavior from observations.
- Install Ollama and download `qwen3:0.6b` if needed. The warmup runs the required
  command and saves genuine output to `assignments_05/ollama_output.txt`. Paste
  that output into the marked multiline comment in `warmup_05.py`, then add your
  observed comparison with the printed OpenAI response.
- Check the live temperature/persona/sentiment results and update comparison
  comments with your observations where appropriate.
- Chat for 3–4 turns, give your name, and later ask the bot to recall it. Try both
  features and confirm quit exits cleanly. Feature results are also kept in history.
- Review the ethics reflection and separate mindset reflection in your own voice.

## Verification

Offline tests cover history across turns, feature routing, moderation of feature
inputs, malformed JSON, and quitting inside a nested prompt. These tests use fake
responses and do not establish actual model quality or live moderation behavior.

```console
python -m unittest discover -s assignments_05 -p "test_*.py" -v
```

AI assistance was used to draft the implementation and reflection. Review and
understand the code before submitting, and follow your course's disclosure rules.

API calls follow the official [Chat Completions reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)
and [moderation guide](https://developers.openai.com/api/docs/guides/moderation),
while retaining the models and parameter exercises required by the course.
