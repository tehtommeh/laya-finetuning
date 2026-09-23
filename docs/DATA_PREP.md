# Preparing fine-tuning data

How to turn the data you already have (helpdesk exports, inboxes, logs, annotation sheets) into the JSONL that
`make validate` and `make train` expect. The format itself is specified in [FINETUNING.md](FINETUNING.md#data-format).
This guide is about getting there.

- [The target](#the-target)
- [Step 1: fix the schema first](#step-1-fix-the-schema-first)
- [Step 2: collect states](#step-2-collect-states)
- [Step 3: get labels](#step-3-get-labels)
- [Step 4: clean](#step-4-clean)
- [Step 5: split without leakage](#step-5-split-without-leakage)
- [Step 6: validate and iterate](#step-6-validate-and-iterate)
- [Recipes](#recipes): CSV export · annotator votes · LLM teacher · time/group split · email
- [Checklist](#checklist)

All recipes are plain Python 3 (standard library only). Run them on the host and write into `data/<name>/`, which
the training container sees as `/data/<name>/`.

## The target

```
data/mine/
  questions.json   the schema: {question_id: {type, instructions, criteria}}
  train.jsonl      one case per line: {"id", "state", "gold", optional "workflow"}
  test.jsonl       optional but recommended: held out by time or group (Step 5)
  calib.jsonl      optional: otherwise 10% of train is held out for calibration
```

```json
{"id": "tkt-10482", "state": {"subject": "Charged twice", "body": "..."}, "gold": {"team": "billing", "urgency": 2, "refund_requested": true}}
```

## Step 1: fix the schema first

Write `questions.json` **before** collecting labels. The question wording and the option descriptions are part of
the model's input, and every label you collect is an answer to that exact wording.

- Decompose the decision into small questions (see [USE_CASES.md](USE_CASES.md#combinations-several-small-questions-one-pass)):
  `team` + `urgency` + `needs_human` rather than one 12-option label.
- Give every `choice` an escape option (`other`) and one-line descriptions.
- Make `score` levels short, ordered and distinct.
- Write the labelling guideline next to it: one example per option, plus how to decide borderline cases. The
  same text makes good `criteria` descriptions.
- Changing the schema later means relabelling, or at least re-checking, the affected question.

## Step 2: collect states

A state is the input exactly as it will look **in production**, at the moment the decision is made.

- **Same shape as production.** If production sends `{"subject", "body", "from"}`, so should training. If
  production sees only the first message of a thread, train on the first message, not the whole resolved thread.
- **No future information.** Do not put the ticket's final resolution, the agent's reply or the "closed as" field
  in the state. That is leakage: the model learns to read the answer, and then fails in production.
- **Sample from real traffic,** including the messy, ambiguous and rare cases. Stratify if one category is 90% of
  volume. Keep a natural sample for the test set, and optionally oversample rare labels in train.
- **Structured is fine.** A JSON object with named fields (`{"plan": "pro", "tenure_months": 14, "message": "..."}`)
  lets the model use metadata. Keep keys stable across cases.
- **Size.** Measure a few with `make validate` (it reports token lengths). The English base leaves ~320–500 tokens
  for the state, and fine-tunes at the default `--max-len 1024` leave ~770. Put the important fields first, because
  overflow is cut from the end.

Where states usually come from: helpdesk exports (Zendesk, Freshdesk, Jira), email archives (mbox, Gmail
exports), chat logs, CRM notes, application logs, agent traces, form submissions, database tables.

## Step 3: get labels

In rough order of cost:

| source | what it is | label quality | notes |
|---|---|---|---|
| **Existing outcomes** | fields your systems already record: the team a ticket was finally assigned to, whether a refund was issued, the severity an engineer set | medium: noisy but free | map them to your schema; check the field meant what you think it meant at the time |
| **LLM teacher** | ask a strong LLM your exact questions, N times per case | medium–good, soft | cheap at scale; sample N ≥ 5 and use the vote split as soft labels (recipe below) |
| **Single annotator** | a person answers each question | good | fast; write down guidelines to keep it consistent |
| **Multiple annotators** | 2–5 people per case | best, soft | vote splits become soft labels; disagreement shows where your schema is ambiguous |

Mixing sources works: LLM-teacher labels for volume, plus a human-labelled test set to judge by. **Always build
the test set from the most trusted labels you have**, because the test set is what you make decisions from.

How much: see [FINETUNING.md § How much data](FINETUNING.md#how-much-data). On the benchmark, gains flattened at
~150 cases per question set, and 20–40 was a useful pilot.

## Step 4: clean

- **Deduplicate** exact and near-exact states (templated notifications, forwarded copies). Duplicates across
  train and test inflate the results.
- **Strip boilerplate you would strip in production:** quoted reply chains, signatures, legal footers, tracking
  pixels, HTML. For email, the SDK's `laya.clean_email_body` / `laya.email_state` do exactly what the API would.
- **Normalise whitespace and encodings,** and drop empty or placeholder states ("test", "asdf").
- **Handle personal data deliberately.** Masking names, emails and numbers (`[EMAIL]`, `[PHONE]`) is often fine,
  and required by policy in many places. But mask identically in production, or the model sees a different
  distribution. Don't mask what a question depends on (a `pii` flag needs the PII).
- **Drop cases whose label you can't trust** (auto-closed tickets, reassigned three times, annotators split 50/50
  on a hard label) rather than guessing. Or keep them as honest soft labels.

## Step 5: split without leakage

`train.py` can carve calibration and test sets out of `train.jsonl` by hashing `id`. That's fine for independent
cases, but it leaks when cases are related. Write an explicit `test.jsonl` when:

- **Cases share a thread, customer or document** (several messages from one conversation). Put whole groups on one
  side, or the model "recognises" the customer rather than learning the task.
- **Traffic changes over time.** Hold out the most recent weeks as test. That is closest to what production will
  see, and it catches drift.
- **Labels came from different sources.** Keep the human-labelled cases as test, and train on the rest.

Keep the test set untouched once you start comparing runs. If you tune on it, it stops measuring anything.

## Step 6: validate and iterate

```bash
make validate DATA=/data/mine
```

It reports parse errors with `file:line`, per-question label counts, dominant labels (> 90%), questions with a
single label, soft-label counts, sequence lengths and truncation, and options that don't fit the token budget.
Fix every ERROR and read every WARN. Then run a small pilot (`make train DATA=/data/mine RUN=pilot ARGS="--limit 100
--epochs 12"`), `make evaluate RUN=pilot`, and look at the **per-question** results in `runs/pilot/eval.json`. The
weakest question is where the next labels should go.

## Recipes

### A. Helpdesk / CSV export with existing outcome fields

`tickets.csv` has `ticket_id, created_at, subject, body, final_group, priority, refund_issued`. Map the existing
fields onto your schema, and keep only what was known when the ticket arrived in the state.

```python
#!/usr/bin/env python3
"""tickets.csv -> data/mine/train.jsonl, using existing outcome fields as labels."""
import csv, json, os

TEAM = {"Billing": "billing", "Payments": "billing", "Tech Support": "technical",
        "Engineering": "technical", "Sales": "sales"}               # everything else -> "other"
URGENCY = {"low": 0, "normal": 1, "high": 2, "urgent": 3}           # score level index

os.makedirs("data/mine", exist_ok=True)
seen, n = set(), 0
with open("tickets.csv", newline="", encoding="utf-8") as f, \
     open("data/mine/train.jsonl", "w", encoding="utf-8") as out:
    for row in csv.DictReader(f):
        body = " ".join(row["body"].split())                      # normalise whitespace
        if len(body) < 15 or (row["subject"], body) in seen:      # drop empties and exact duplicates
            continue
        seen.add((row["subject"], body))
        gold = {"team": TEAM.get(row["final_group"], "other")}
        if row["priority"].lower() in URGENCY:
            gold["urgency"] = URGENCY[row["priority"].lower()]
        if row["refund_issued"] in ("true", "false"):
            gold["refund_requested"] = row["refund_issued"] == "true"   # a proxy: check it matches the question
        case = {"id": row["ticket_id"], "created_at": row["created_at"],   # extra keys are ignored by training
                "state": {"subject": row["subject"], "body": body},       # only what was known at arrival
                "gold": gold}
        out.write(json.dumps(case, ensure_ascii=False) + "\n")
        n += 1
print("wrote", n, "cases")
```

Watch out for proxies: "a refund was issued" is not "the customer asked for a refund". Spot-check 50 cases by hand
before trusting a mapped field.

### B. Aggregating several annotators' votes into soft labels

`labels.csv` is one row per (case, annotator, question): `case_id, annotator, question, answer`. States come
from a separate `cases.jsonl` (`{"id", "state"}`).

```python
#!/usr/bin/env python3
"""Per-annotator answers -> soft labels. Vote counts go straight into `probabilities`
(training normalises them), so 2 of 3 annotators saying billing = 0.67 billing."""
import csv, json
from collections import defaultdict

votes = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))   # case -> question -> answer -> count
with open("labels.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        votes[r["case_id"]][r["question"]][r["answer"].strip().lower()] += 1

MIN_VOTES = 2                                   # need at least 2 opinions to trust a soft label
with open("cases.jsonl", encoding="utf-8") as f, open("data/mine/train.jsonl", "w", encoding="utf-8") as out:
    for line in f:
        case = json.loads(line)
        gold = {q: {"probabilities": dict(ans)} for q, ans in votes[case["id"]].items()
                if sum(ans.values()) >= MIN_VOTES}
        if gold:
            out.write(json.dumps({"id": case["id"], "state": case["state"], "gold": gold},
                                 ensure_ascii=False) + "\n")
```

Answers must be option names exactly as in `questions.json`: labels for `choice`, `"0"`–`"K-1"` for `score`,
`"true"`/`"false"` for `noul`. `make validate` names any answer that doesn't match. Questions where annotators
often split 50/50 usually need a clearer schema, not more votes.

### C. LLM teacher with repeated sampling

Ask a strong LLM the *same* questions, several times per case at temperature ~1, and use the answer counts as a
soft label. This is how the typed-decisions benchmark's gold labels were made (teacher samples). The provider call
is left to you: `ask_llm` must return one answer per question, as option names.

```python
#!/usr/bin/env python3
"""Soft labels from N teacher samples per case."""
import json
from collections import Counter

N = 7
QUESTIONS = json.load(open("data/mine/questions.json"))

def options(q):
    if q["type"] == "choice":
        return list(q["criteria"])                      # dict keys or a plain list of labels
    if q["type"] == "score":
        return [str(i) for i in range(len(q["criteria"]))]
    return ["false", "true"]

def prompt(state):
    lines = ["Answer each question about the input. Reply with JSON {question_id: option}.",
             "Input:", json.dumps(state, ensure_ascii=False), "", "Questions:"]
    for qid, q in QUESTIONS.items():
        opts = q["criteria"] if q["type"] != "noul" else {"true": "yes", "false": "no"}
        if q["type"] == "score":
            opts = {str(i): d for i, d in enumerate(q["criteria"])}
        lines.append("- %s: %s Options: %s" % (qid, q["instructions"], json.dumps(opts, ensure_ascii=False)))
    return "\n".join(lines)

def ask_llm(text):
    """Call your LLM provider with `text`; return a dict {question_id: option_name}."""
    raise NotImplementedError

with open("cases.jsonl", encoding="utf-8") as f, open("data/mine/train.jsonl", "w", encoding="utf-8") as out:
    for line in f:
        case = json.loads(line)
        counts = {qid: Counter() for qid in QUESTIONS}
        for _ in range(N):
            try:
                answers = ask_llm(prompt(case["state"]))
            except Exception:
                continue                                    # skip a failed sample, keep the others
            for qid, a in answers.items():
                if qid in QUESTIONS and str(a).lower() in [o.lower() for o in options(QUESTIONS[qid])]:
                    counts[qid][next(o for o in options(QUESTIONS[qid]) if o.lower() == str(a).lower())] += 1
        gold = {qid: {"probabilities": dict(c)} for qid, c in counts.items() if sum(c.values()) >= N // 2 + 1}
        if gold:
            out.write(json.dumps({"id": case["id"], "state": case["state"], "gold": gold},
                                 ensure_ascii=False) + "\n")
```

Keep a human-labelled test set when training on teacher labels. Otherwise you measure agreement with the teacher,
not correctness. Also check the provider's terms allow using outputs to train another model.

### D. Splitting by time or by group

```python
#!/usr/bin/env python3
"""all.jsonl -> train.jsonl + test.jsonl: the newest ~15% as test, never splitting a group.
A group (thread, customer, document) is placed by when it *started*, so a long-running thread
cannot drag old cases into the test set."""
import json

GROUP_KEY = "thread_id"          # cases sharing this value stay on one side; None = split cases individually
cases = [json.loads(l) for l in open("data/mine/all.jsonl", encoding="utf-8")]

def group(c):
    return c.get(GROUP_KEY) if GROUP_KEY and c.get(GROUP_KEY) is not None else "case:" + c["id"]

start = {}
for c in cases:                                   # each group's first timestamp
    g = group(c)
    start[g] = min(start.get(g, c["created_at"]), c["created_at"])
starts = sorted(start.values())
cut = starts[int(len(starts) * 0.85)]             # groups starting in the newest 15% go to test

train = [c for c in cases if start[group(c)] < cut]
test = [c for c in cases if start[group(c)] >= cut]
for name, rows in (("train", train), ("test", test)):
    with open("data/mine/%s.jsonl" % name, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(c, ensure_ascii=False) + "\n" for c in rows)
print("train", len(train), "test", len(test))
```

With an explicit `test.jsonl`, `train.py` still carves 10% of train for calibration (`--calib-frac`), and uses
your test file for evaluation.

### E. Email archives

The API sees whatever you send, so clean training emails the same way production will. The SDK's cleaner strips
quoted threads, signatures and disclaimers. Run it in the training container, where `laya` is installed:

```bash
docker compose --profile train run --rm --entrypoint python3 train -c '
import json, laya
with open("/data/mine/raw_emails.jsonl") as f, open("/data/mine/cases.jsonl", "w") as out:
    for line in f:
        e = json.loads(line)
        state = laya.email_state(e["subject"], e["body"], sender=e.get("from"))
        out.write(json.dumps({"id": e["id"], "state": state}, ensure_ascii=False) + "\n")
'
```

Then label `cases.jsonl` with recipe B or C, and use `laya.email_state` on the serving path too.

## Checklist

- [ ] `questions.json` written and frozen, each `choice` with an `other` option, labelling guideline written
- [ ] states look exactly like production input, at decision time, with no answer-revealing fields
- [ ] duplicates removed, boilerplate stripped the same way production strips it, PII handled consistently
- [ ] labels mapped to exact option names; proxy fields spot-checked by hand
- [ ] every option appears in train; rare options oversampled if needed; nothing above ~90% dominance
- [ ] test set held out by time or group, labelled by your most trusted source, and never tuned on
- [ ] `make validate` shows 0 errors, WARNs understood, truncation under ~5%
- [ ] a pilot run evaluated, and the per-question results read, before labelling at scale
