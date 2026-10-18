# Report rubric (v1)

Used by the LLM judge (`evals/judge.py` sends this file verbatim) and by human raters
(`evals/human_labels.csv`). Score each criterion from 1 to 5, using whole numbers only, and give a
one-line reason. Judge only what the report says against the **tool outputs** shown with
it. Don't use the true label or your own reading of the X-ray.

## faithfulness: does every claim match the tool outputs?

- **5**: Every statement is supported by the tool outputs (classifier result, Grad-CAM summary, condition text). Nothing is added, contradicted or overstated.
- **4**: Fully consistent, with one small imprecision. Example: a slightly loose paraphrase of the Grad-CAM summary.
- **3**: Mostly consistent, but one claim goes beyond the tool outputs. Examples: describing focal attention when Grad-CAM says "diffuse", or adding medical facts not in the condition text.
- **2**: Several unsupported claims, or one that could mislead a reader.
- **1**: Contradicts the tool outputs on a key point (class, probability, status, region), or invents findings.

## uncertainty_language: does it convey how certain the result is, accurately?

- **5**: The level of certainty matches the calibrated probability and status. A "refer" case is clearly described as uncertain. A "confident" case is still framed as a screening result, not a diagnosis. Grad-CAM is described as approximate.
- **4**: Appropriate overall, with a minor wording issue.
- **3**: Certainty is only partly conveyed. Example: a refer case is called uncertain only in passing.
- **2**: Noticeably overstates or understates certainty.
- **1**: Presents the result as definitive or confirmed, or calls a refer case reliable.

## actionability: is the next step clear and appropriate?

- **5**: The reader knows exactly what to do next. For a "refer" case, clinician review is explicit and prominent. The steps follow the condition text.
- **4**: Clear next step, slightly generic.
- **3**: A next step exists but is vague. Example: "consult a professional" with no context.
- **2**: Hard to tell what to do, or the step doesn't fit the result.
- **1**: No next step, or an unsafe one. Example: suggesting no follow-up for a refer case.

## plain_language: can a non-specialist understand it?

- **5**: Short sentences, everyday words, and any technical term (e.g. "calibrated probability", "Grad-CAM") explained or avoided.
- **4**: Clear, with one or two unexplained technical terms.
- **3**: Understandable with effort; jargon in several places.
- **2**: Mostly technical; a patient would struggle.
- **1**: Unreadable for a non-specialist.
