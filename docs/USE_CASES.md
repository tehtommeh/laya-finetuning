# Use cases

Ideas for what to build with Laya's three question types, how to combine them, and patterns for fitting Laya
into a larger system. Question-type mechanics (output fields, limits, token budgets) are in the
[README](../README.md#question-types). Fine-tuning a schema is in [FINETUNING.md](FINETUNING.md).

- [Choosing a type](#choosing-a-type)
- [`noul`: yes/no flags and gates](#noul-yesno-flags-and-gates)
- [`choice`: pick one path](#choice-pick-one-path)
- [`score`: ordered levels and rubrics](#score-ordered-levels-and-rubrics)
- [Combinations: several small questions, one pass](#combinations-several-small-questions-one-pass)
- [Worked question sets](#worked-question-sets)
- [Hybrid patterns](#hybrid-patterns)
- [Ideas by industry](#ideas-by-industry)
- [Anti-patterns](#anti-patterns)
- [Where Laya does not fit](#where-laya-does-not-fit)

## Choosing a type

Choose by **what you will do with the answer**, not by how the question is phrased.

| if the answer… | use | because |
|---|---|---|
| triggers an action or an if-statement | `noul` | one probability, and a threshold you can tune for precision or recall |
| picks exactly one path out of several | `choice` | the options compete, and the probabilities sum to 1 |
| sits on a scale where "close" counts | `score` | the expected value ranks and averages, and off-by-one beats off-by-three |
| could be several things at once | several `noul`s | `choice` would force one label and lose the rest |
| needs a reason, a number read out of the text, or free text | not Laya | see [below](#where-laya-does-not-fit) |

Quick tests:

- **Are the options mutually exclusive?** No → one `noul` per option. Yes → `choice`.
- **Is "medium" between "low" and "high"?** Yes → `score`. If the labels are just different buckets → `choice`.
- **Will code branch on it?** Usually a `noul`, even when the underlying concept feels like a category.

## `noul`: yes/no flags and gates

The most accurate type after fine-tuning (0.82 on the benchmark), and the easiest to operate. After hard-label
calibration, P(true) = 0.9 means right about 90% of the time, so a threshold is a business decision, not a guess.

**Safety and trust**
- Jailbreak or prompt-injection check before input reaches your LLM
- Does this LLM *output* leak the system prompt, secrets or another user's data?
- Harassment, threats or self-harm signals in chat and community posts
- Is this review fake, incentivised or off-topic?
- Scam or phishing detection on inbound email, SMS and marketplace messages
- Is this account signup likely a bot (from the profile text and first message)?
- Is this user impersonating staff or a brand?
- Does this image caption or alt text contain hidden instructions (indirect prompt injection in RAG content)?

**Email and communication**
- Does this email need a reply?
- Is the sender asking a question, or just informing?
- Out-of-office or auto-reply detection, so automations don't respond to robots
- Does this message contain a commitment ("I'll send it Friday")? Feed a task tracker.
- Is this a meeting request?
- Is the sender upset, even if politely worded?
- Does this thread need a manager's attention?

**Compliance and review**
- Does this contract clause include auto-renewal, indemnity, exclusivity or a non-compete?
- Does this sales email promise pricing, discounts or delivery dates?
- Does this document contain personal data (names, addresses, health details, ID numbers)?
- Does this commit message, log line or ticket mention a credential or secret?
- Does this marketing copy make a claim that needs legal review ("guaranteed", "clinically proven")?
- Does this expense description breach policy (alcohol, gifts over the limit, personal items)?
- Does this chat transcript show the agent giving financial, legal or medical advice they shouldn't?

**Pipeline gates (cheap checks before expensive work)**
- Is this a real support request, or spam or a newsletter?
- Is this document in scope for the pipeline at all?
- Did this AI agent's step violate a stated constraint?
- Is this question answerable from our docs, or should it go to a human?
- Has the user already provided enough information to proceed?
- Is this a duplicate of the issue described in the previous message?
- Does this retrieved passage actually answer the question (a RAG relevance filter)?

**Product and UX signals**
- Is the user confused about how a feature works?
- Is this feedback a feature request?
- Does the user mention a competitor?
- Is the user asking to be contacted by a human?
- Is the user trying to cancel?

## `choice`: pick one path

The options compete, so use it when exactly one answer is right. Keep it under ~20 options, and always include an
escape option (`other`, `none of the above`).

**Routing**
- Which team or queue gets this ticket
- Which LLM handles this prompt: small, large, code-specialised or refuse (the SDK's `router` preset does this)
- Which knowledge base or index to search for this question
- Which regional team handles this enquiry
- Which tool an agent should call next (`search`, `calculator`, `calendar`, `none`)
- Which prompt template or persona fits this request

**Categorisation for analytics**
- Reason for cancellation, from churn surveys
- Feedback theme: pricing, UX, performance, missing feature, support
- Bug report component: frontend, backend, billing, auth, mobile
- Lead source or buyer intent from inbound forms
- Return reason: damaged, wrong item, didn't fit, changed mind, late
- Call outcome from a sales-call summary: booked, follow-up, lost, not a fit
- Topic of a community post, for tagging and discovery

**Workflow steps**
- Next action for an agent: answer, ask for more information, escalate, close
- Invoice disposition: approve, hold, reject, manual review
- Security alert: close as benign, investigate, contain
- Content moderation outcome: allow, label, restrict, remove
- Refund decision: approve automatically, partial refund, send to a human
- Claims processing: pay, request documents, investigate, deny
- Onboarding step the user is stuck on

**Detection and classification**
- Language register: formal, casual, legal, technical
- Document type: invoice, receipt, contract, CV, letter, other
- Message intent: question, complaint, praise, request, spam
- Which product or plan the customer is talking about
- Speaker role in a transcript: customer, agent, bot

## `score`: ordered levels and rubrics

Use `score` for averages, rankings and dashboards, and take the most likely level when you need one mark. It's the
weakest type for exact levels (0.68 on the benchmark), but usually close (mean error 0.32 levels). 3–5 short levels
work best.

**Prioritisation**
- Ticket urgency: sort the queue by the expected score
- Bug severity from the report text
- Incident impact: one user, a team, a region, everyone
- Lead quality or deal readiness
- How likely a user is to churn, from their latest message
- Fraud risk level of a transaction description

**Rubric marking**
- Grading support replies on accuracy, completeness and tone (one score each)
- Marking student short answers against a rubric
- Code review comment severity: nitpick, suggestion, should fix, must fix
- Quality of LLM outputs as a fast automated evaluation, an alternative to an LLM-as-judge for high volume
- Essay feedback: structure, argument, evidence, clarity
- Interview answer quality against a competency framework
- Documentation quality: missing, stub, adequate, excellent

**Intensity and degree**
- Customer frustration level, for proactive outreach
- Sentiment from very negative to very positive, tracked over time
- Risk level of an AI agent's proposed action, checked before it executes
- How technical a question is, to route it to a junior or senior responder
- Formality or politeness of a message (tone checking before sending)
- Reading difficulty of a passage
- How complete a bug report is (0 = nothing reproducible, 3 = steps, expected, actual, environment)
- Confidence that a lead matches the ideal customer profile

## Combinations: several small questions, one pass

Five questions cost barely more than one (23 ms for 1, 24 ms for 5, 35 ms for 10 on an RTX 3090), so decompose
the decision instead of cramming it into one question. Each piece stays simple and learnable, you can inspect it
on its own, and changing policy means changing code rather than relabelling data.

| use case | `noul` | `choice` | `score` | your code decides |
|---|---|---|---|---|
| Support triage | `refund_requested`, `churn_risk`, `needs_human` | `team` | `urgency` | queue, priority, SLA |
| Inbox assistant | `needs_reply`, `is_newsletter` | `category` | `importance` | archive, file or surface |
| LLM guardrail layer | `jailbreak`, `injection`, `pii` | `topic` | `harm_severity` | allow, refuse or redact |
| Agent observability | `needs_review`, `constraint_violated` | `outcome` | `risk` | pass, sample or page a human |
| Sales email review | `commits_pricing`, `mentions_competitor` | `deal_stage` | `buyer_interest` | CRM updates, coaching flags |
| Hiring pipeline | `meets_requirements` | `best_fit_role` | `seniority` | shortlist for a human, never auto-reject |
| Content pipeline | `on_brand`, `factual_claims_present` | `content_type` | `quality` | publish, edit or reject |
| Invoice processing | `duplicate`, `matches_order` | `disposition` | `discrepancy_severity` | pay, hold, escalate |
| Security alerts | `true_positive`, `credential_compromise` | `disposition` | `severity` | close, ticket or page on-call |
| Community moderation | `toxic`, `spam`, `threat` | `violation_type` | `severity` | allow, warn, hide or ban |
| Code review bot | `security_relevant`, `breaking_change` | `comment_type` | `severity` | block the merge or leave a note |
| Product feedback | `feature_request`, `bug_report`, `mentions_competitor` | `product_area` | `sentiment` | roadmap tagging, bug filing |
| Returns desk | `within_policy`, `fraud_signals` | `return_reason` | `customer_value` | auto-approve, review or deny |
| IT helpdesk | `outage_related`, `security_incident` | `system` | `user_impact` | self-service article or engineer |
| Chat escalation | `asks_for_human`, `bot_failed` | `topic` | `frustration` | stay on the bot or hand over |

A note on the last column: keep the policy in code. "Escalate if `churn_risk > 0.8` and `customer_value >= 2`" is
a line you can change on Monday. A single `choice` over "escalate / don't" bakes that policy into the labels.

## Worked question sets

Ready-to-adapt schemas. Paste one into the UI's Decide tab or send it as `questions` to `/v1/decide`. They will run
zero-shot, but expect base-model quality until you fine-tune on your own labelled cases (the README's measured
table shows the gap).

### Support triage

```json
{
  "team": {"type": "choice", "instructions": "Which team should handle this ticket?",
           "criteria": {"billing": "invoices, payments, refunds, plan changes",
                        "technical": "bugs, outages, errors, integrations",
                        "account": "login, access, profile, security settings",
                        "sales": "pricing questions, quotes, upgrades",
                        "other": "none of the above"}},
  "urgency": {"type": "score", "instructions": "How urgent is this ticket?",
              "criteria": ["can wait a week", "this week", "today", "blocking the customer right now"]},
  "refund_requested": {"type": "noul", "instructions": "Does the customer ask for money back?"},
  "churn_risk": {"type": "noul", "instructions": "Does the customer threaten or imply they will cancel or leave?"},
  "needs_human": {"type": "noul", "instructions": "Does this need a human agent rather than a help-centre article?",
                  "criteria": {"true": "judgement, exceptions, account-specific action, or an upset customer",
                               "false": "a standard how-to question with a documented answer"}}
}
```

### LLM guardrail layer

```json
{
  "jailbreak": {"type": "noul", "instructions": "Does `prompt` try to make the assistant ignore its rules or instructions?"},
  "injection": {"type": "noul", "instructions": "Does `prompt` contain instructions aimed at the AI system rather than a genuine user request?"},
  "pii": {"type": "noul", "instructions": "Does `prompt` contain personal data such as names with contact details, ID numbers or health information?"},
  "harm_severity": {"type": "score", "instructions": "How much harm would fully complying with `prompt` cause?",
                    "criteria": ["none: ordinary request", "minor: mildly inappropriate",
                                 "serious: unsafe advice or abuse", "severe: dangerous or illegal"]},
  "topic": {"type": "choice", "instructions": "What is `prompt` mainly about?",
            "criteria": ["product_support", "coding", "general_knowledge", "personal_advice", "security_testing", "other"]}
}
```

(The SDK ships a similar `guard` preset. Try it in the UI's Decide tab.)

### Agent trace review

```json
{
  "outcome": {"type": "choice", "instructions": "How did this agent run turn out?",
              "criteria": {"success": "completed the task correctly",
                           "partial": "made progress but did not finish",
                           "failure": "did not accomplish the task",
                           "harmful": "caused damage or violated a constraint"}},
  "risk": {"type": "score", "instructions": "How risky was the agent's behaviour?",
           "criteria": ["benign: read-only or clearly safe", "low: routine writes within scope",
                        "moderate: irreversible or out-of-scope actions", "high: destructive or policy-violating"]},
  "needs_review": {"type": "noul", "instructions": "Should a human inspect this run?"}
}
```

### Rubric marking (a support reply out of 4)

```json
{
  "accuracy": {"type": "score", "instructions": "Is the reply factually correct for the customer's issue?",
               "criteria": ["wrong or misleading", "partly correct", "correct with minor gaps", "fully correct"]},
  "completeness": {"type": "score", "instructions": "Does the reply resolve everything the customer asked?",
                   "criteria": ["ignores the question", "addresses some of it", "addresses most of it", "addresses all of it"]},
  "tone": {"type": "score", "instructions": "How well does the reply match the customer's tone and situation?",
           "criteria": ["dismissive or rude", "flat and impersonal", "polite", "warm and empathetic"]},
  "should_send": {"type": "noul", "instructions": "Is this reply ready to send without edits?"}
}
```

Combine per-criterion scores with your own weights, e.g. `0.5*accuracy + 0.3*completeness + 0.2*tone`, and use
`should_send` as the gate.

### Inbox assistant

```json
{
  "needs_reply": {"type": "noul", "instructions": "Does the sender expect a reply from me?"},
  "is_newsletter": {"type": "noul", "instructions": "Is this a newsletter, marketing or automated notification?"},
  "category": {"type": "choice", "instructions": "What is this email about?",
               "criteria": ["scheduling", "request_for_work", "fyi_update", "billing", "personal", "other"]},
  "importance": {"type": "score", "instructions": "How important is this email to act on?",
                 "criteria": ["ignore", "low", "normal", "high"]}
}
```

## Hybrid patterns

Laya is at its best as the fast, calibrated decision layer inside a larger system.

**Confidence-gated LLM fallback.** Laya answers everything. Cases below a confidence threshold go to an LLM or a
human. Calibrated probabilities make the split trustworthy, and you choose the threshold from the error rate you
can accept. Typically most traffic clears the gate, so the expensive path handles only the ambiguous cases.

**Pre-filter for expensive calls.** A `noul` ("is this worth processing?" or "is this in scope?") in front of a
large-model step. At ~22 ms, the check is nearly free compared with what it saves.

**Decide, then explain.** Laya makes the decision, and an LLM writes the customer-facing explanation or reply, with
the decision given as input. The decision stays consistent and auditable, and the LLM does what it's good at.

**Real-time scoring at volume.** Score every chat message, log line, transaction or event-stream item, where an
LLM per item would be too slow or too expensive. Batch through `/v1/decide/batch`: on the RTX 3090 that is
~2.4 ms per short message (~425/s) with three questions each, 6–9× faster than one call per item. Many concurrent
producers can also just call `/v1/decide`: the API coalesces them (~300 short requests/s at 64+ clients).

**Drift and trend monitoring.** Track the mean `score` or the `choice` distribution per day. A shift (urgency
creeping up, a new category share) is an early signal that traffic changed, before anyone reads a ticket.

**Active labelling.** Send annotators the model's *least* confident cases first. Those labels improve a fine-tune
fastest, and the learning curve in [EXPERIMENTS.md](EXPERIMENTS.md) shows gains up to ~150 cases per question set.

**Coarse to fine.** For large label sets, a first `choice` picks the family ("billing / technical / account"), and
a second `choice`, with only that family's labels, picks the leaf. Each stays under the ~20-option comfort zone.
(The SDK's `shortlist_choice` does a similar job with embeddings.)

**Multi-label fan-out.** Need "which of these apply?" Ask one `noul` per label in the same request, then take
everything above your threshold. It's the same cost as one question, up to ~5 labels.

**Shadow comparison.** Run a new fine-tune alongside the current model via `/v1/compare`. Log disagreements and
review them before switching. The UI's Compare tab does this interactively.

**Language-aware pipelines.** Let the router send non-English text to the multilingual checkpoint automatically,
or fine-tune the multilingual base once and serve every language with one model (it came within a point of the
English base on English data, at twice the speed).

**Human-in-the-loop threshold tuning.** Start with a conservative threshold (auto-act only above 0.95). Log what
humans decide on the rest, and lower the threshold as the logged agreement rate proves it's safe.

**Agent guardrails.** Before an agent executes a tool call, score the proposed action (`risk`, `irreversible`,
`within_scope`) and require confirmation above a level. It's cheap enough to run on every step.

**Evaluation at scale.** Use rubric `score`s to evaluate thousands of LLM or agent outputs per minute for
regressions. Spot-check with a slower LLM judge or humans, rather than running the judge on everything.

## Ideas by industry

**E-commerce and retail:** return reason (`choice`); delivery issue flag (`noul`); product question vs complaint
(`choice`); review sentiment and helpfulness (`score`); fraud signals in order notes (`noul`); size or fit feedback
theme (`choice`).

**Finance and fintech:** transaction description category (`choice`); expense policy breach (`noul`); dispute
reason (`choice`); complaint severity for regulatory reporting (`score`); vulnerable-customer signals (`noul`);
KYC document type (`choice`).

**Insurance:** claim type (`choice`); documentation complete (`noul`); fraud-indicator level (`score`); next step:
pay, request info, investigate (`choice`); customer distress (`score`).

**Healthcare admin (routing and triage of admin requests, not clinical decisions):** request type such as
appointment, prescription refill, billing or records (`choice`); urgency of the admin request (`score`); mentions
of symptoms that need a clinician's review (`noul`); personal data present (`noul`).

**Legal and compliance:** clause type (`choice`); risky clause present (`noul`); contract risk level (`score`);
matter intake routing by practice area (`choice`); privilege-sensitive content (`noul`).

**HR and recruiting:** leave request type (`choice`); policy question vs grievance (`choice`); CV meets minimum
requirements (`noul`); seniority estimate (`score`); interview note sentiment (`score`); harassment report flag
(`noul`).

**Education:** short-answer marking against a rubric (`score`); question type such as conceptual, procedural or
admin (`choice`); student needs help (`noul`); academic-integrity flags for review (`noul`); essay criteria marks
(`score` per criterion).

**Software and DevOps:** log line severity (`score`); alert is actionable (`noul`); incident type (`choice`); bug
report completeness (`score`); PR comment severity (`score`); breaking-change mention (`noul`); on-call page vs
ticket (`choice`).

**Security operations:** alert true positive (`noul`); disposition: close, investigate, contain (`choice`);
severity (`score`); credential compromise indicated (`noul`); phishing report triage (`choice`).

**Marketplaces and communities:** listing violates policy (`noul`); listing category (`choice`); scam message
between users (`noul`); toxicity severity (`score`); off-platform payment request (`noul`).

**Travel and hospitality:** booking change type such as date, cancel or upgrade (`choice`); complaint severity
(`score`); refund eligibility signals (`noul`); VIP handling needed (`noul`).

**Media and publishing:** article topic (`choice`); needs fact-check (`noul`); headline clickbait level (`score`);
reader comment moderation (`noul` + `score`).

**Sales and marketing:** lead intent (`choice`); buying-signal strength (`score`); competitor mentioned (`noul`);
unsubscribe or angry-reply detection (`noul`); campaign reply category (`choice`).

**Games and social apps:** toxic chat (`noul`); cheating or exploit report (`noul`); player feedback theme
(`choice`); frustration level before churn (`score`).

## Anti-patterns

- **`choice` for multi-label questions.** "Which issues does this mention?" forces one label. Use one `noul` per
  label.
- **`choice` for ordered levels.** "low / medium / high" as a `choice` throws away the order. Use `score`.
- **One giant question.** A single `choice` over "urgent-billing / normal-billing / urgent-tech / …" multiplies the
  labels and bakes policy into data. Decompose it into `team` + `urgency` and decide in code.
- **Compound `noul`s.** "Is this urgent and about billing?" hides which part was true. Ask two questions.
- **Vague questions.** "Is this about money?" is ambiguous; "Does the customer ask for money back?" is learnable.
- **No escape option.** A `choice` without `other` confidently mislabels everything outside your categories.
- **Long rubric levels.** Paragraph-long `score` descriptors get truncated in the option budget. Keep each level
  to one line.
- **Rewording after fine-tuning.** The question wording is part of the model's input. Changing it moves you away
  from what the model learned.
- **Trusting shipped probabilities as-is.** Base checkpoints are over-confident. Calibrate (fine-tunes do this
  automatically) before thresholding.
- **Using `confidence` on `choice` as P(correct).** It's entropy-based. Threshold `probabilities[choice]` instead.
- **Letting it decide alone where the stakes are high.** Hiring, credit, medical, legal: use Laya to sort, flag and
  prioritise for a human, not to make the final call.

## Where Laya does not fit

- **Explanations.** It never says *why*. Pair it with an LLM when a reason is needed.
- **Extraction.** It can say "does this invoice have a PO number?" (`noul`) but cannot read the number out. Use
  an extractor or an LLM for values.
- **Open-ended generation.** Replies, summaries and rewrites are LLM work. Laya can decide *which* template or
  *whether* to reply.
- **Long documents.** A state is cut after ~320–770 tokens, depending on checkpoint and question. Chunk long
  documents and ask per chunk, or summarise first.
- **Multi-step reasoning or arithmetic** over the input ("is the invoice total equal to the sum of the lines?").
  Compute it in code and pass the result in the state instead.
- **Huge label spaces** (hundreds of classes), unless you use coarse-to-fine or shortlisting.
