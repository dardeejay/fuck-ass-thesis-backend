# Project Handa — Your Results, in Plain Language

This walks through Chapter 3's analysis without the jargon, using your actual
numbers. Read this to *understand* the results; use `DEFENSE_FLOW.md` for how to
*present* them. Nothing here contradicts the paper — it's the same findings, translated.

---

## 1. What were we actually trying to find out?

In plain terms: **can a computer program learn, from three years of past weather
data, which kinds of days in Olongapo City tend to end in a class suspension — and
then guess correctly on a new day it's never seen?**

That's it. Everything in Chapter 3 is just different ways of answering "how good is
the program's guess, really?"

---

## 2. The data, in plain terms

- We had **731 days** of weather records (Jan 2023 – Dec 2024).
- Of those, **24 days** (about 1 in 30) were actual class-suspension days.
- The other **707 days** were ordinary days.

Picture a classroom of 731 students where only 24 of them are wearing a red shirt.
That's how rare a "suspension day" is in this data. This rarity is the single most
important thing to understand about everything that follows — it's why some numbers
that *look* impressive (like 95.6% accuracy) are actually less meaningful than they
sound, and why we didn't just check the program's homework once, but multiple times
in different ways.

---

## 3. What is "Random Forest," in plain terms?

Imagine asking **300 different people** to each look at a smaller, random slice of
the weather history and make their own simple rule for "suspend or not" (e.g., "if
it rained a lot the last few days, suspend"). Then you take a vote among all 300, and
whichever answer gets the most votes wins.

That's a Random Forest. Each individual "person" (called a *decision tree*) is a
simple, sometimes-wrong rule-of-thumb; the *voting* across all 300 of them is what
makes the combined guess more reliable than any single rule.

We also tried two other kinds of programs for comparison — **Logistic Regression**
(a program that fits one smooth mathematical curve to the data, no voting) and
**Gradient Boosting** (a different way of combining many simple rules, where each new
rule tries to fix the previous ones' mistakes). Comparing all three is the
"benchmarking" part of the title.

---

## 4. The four scorecards: accuracy, precision, recall, F1

These are just four different ways of grading the program's guesses. Say the program
looked at the 183 days we held back for testing (it never saw these while learning),
and among those 183 days, **6 were real suspension days**. These are the results for the
regularized Random Forest (the version the paper selects — see §5).

| Scorecard | What it's really asking | Your result | What that means literally |
|---|---|---|---|
| **Accuracy** | "Out of all 183 days, how many did it get right (either correctly said suspend, or correctly said don't)?" | 0.934 (93.4%) | It got 171 of 183 days right. Sounds great — but... |
| **Precision** | "When it *said* 'suspend,' how often was it actually right?" | 0.250 (25%) | It cried "suspend" 12 times; only 3 of those were real suspension days (9 were false alarms). |
| **Recall** | "Of the real suspension days, how many did it actually catch?" | 0.500 (50%) | Of the 6 real suspension days, it caught 3 and missed 3. |
| **F1-score** | "A single blended score of precision and recall, so neither can hide a weakness in the other." | 0.333 | Splits the difference between the two above. |

**Why accuracy is the misleading one here:** if the program just said "never
suspend" for every single day, it would *still* score 177/183 = 96.7% accuracy —
higher than our actual model! — while being completely useless. That's the trap of
grading on accuracy when the thing you care about (suspension days) is rare. This is
exactly why the paper leans on precision/recall/F1 instead of bragging about the 95.6%.

---

## 5. "Overfitting" — the program that memorized instead of learning

We noticed something suspicious: when we tested the Random Forest on the *same* data
it had trained on, it scored a perfect 100% on every scorecard. That's not a good
sign — it's like a student who scored 100% on a practice test only because they
memorized the exact questions, not because they understood the subject. Sure enough,
its score dropped sharply on the *new*, unseen test days.

**The fix:** we forced the "voting committee" to keep its rules simpler — each little
rule-of-thumb has to apply to at least 4 similar days, not carve out a special rule
for just 1 or 2 specific days it happened to memorize. This "simplified" version is
called the **regularized** Random Forest, and it's the version that actually ended
up performing best (see §7).

---

## 6. Why we tested it more than once ("cross-validation")

With only 24 suspension days total, one single test run could get lucky or unlucky
depending on which 6 of those 24 days happened to land in the test group. So instead
of trusting one test, we ran two extra rounds of testing:

- **5-fold cross-validation**: split the 731 days into 5 groups, test on each group
  once (using the other 4 groups to teach the program each time), and average the
  5 scores together.
- **Leave-one-out cross-validation (LOO-CV)**: the most thorough version — test on
  *one single day* at a time, 731 times in a row, each time re-teaching the program
  on all the other 730 days. Then combine all 731 individual guesses into one final
  scorecard.

Think of LOO-CV like giving a student 731 pop quizzes instead of one final exam —
much harder to get lucky or unlucky, so the resulting grade is more trustworthy.

---

## 7. The final verdict: which program won, and why

Under the most thorough test (LOO-CV), here's how the five versions compared on F1
(the blended precision+recall score):

| Version | F1-score | Plain-language takeaway |
|---|---|---|
| Logistic Regression | 0.440 | Decent at catching suspensions, but cries false alarm a lot. |
| Random Forest (unrestricted) | 0.457 | Memorizes too much (see §5) — looks OK here but is fragile. |
| Gradient Boosting | 0.436 | Similar issue to unrestricted Random Forest. |
| **Random Forest (regularized)** | **0.545 — best** | The "keep it simple" version. Best overall balance. |
| Gradient Boosting (regularized) | 0.493 | Second best, close behind. |

The winning version — **Random Forest (regularized)** — in plain numbers:
- **Recall = 0.750**: it correctly catches **3 out of every 4** real suspension days.
- **Precision = 0.429**: but when it says "suspend," it's only right **about 4 times
  out of 10** — so it does raise some false alarms.

**In one sentence:** the program is better at *not missing* a real suspension day
than it is at *avoiding false alarms* — which is a reasonable trade-off for this kind
of tool, since missing an actual bad-weather day is worse than an occasional
unnecessary caution.

---

## 8. Which weather clues mattered most?

We asked the program to rank which pieces of information it leaned on most:

1. **Rain over the last 3 days** (biggest clue)
2. **How hot it got that day** — rainy, overcast days stay cooler, so a low
   daytime high is an indirect "stormy day" signal
3. Rain that day alone
4. Rain over the last 7 days (adds almost nothing once the 3-day total is known)
5. How humid it was
6. What month it was
7. Wind speed
8. How cool it got at night (smallest clue)

**In plain terms:** the program cares more about "has it been raining a lot lately"
than "is it raining hard right now." That matches common sense about flooding — the
ground gets saturated over several days, not from one single downpour. This is
probably the most interesting, explainable finding in the whole study, and it's a
good thing to lead with if you want a "so what" answer during your defense.

---

## 9. A second, independent check: real 2025 rainfall

Everything above is trained and tested on 2023–2024. There was a nagging weak spot:
the actual suspension records mostly fall in 2025, and we had *zero* real weather
data for 2025 — so those suspension days had been sitting out of the analysis
entirely, and the whole 2023–2024 result rested on an unverified assumption (that
unlisted days really were quiet days).

We later found a real, official rainfall record for **all 365 days of 2025**
(temperature/humidity/wind still aren't available for that year, just rainfall).
That let us run a second, independent check, entirely separate from the model
above: **does the same "more rain = more likely a suspension" pattern show up in
2025's real data, on days the model never saw during training?**

The answer, in plain numbers:

- Suspended 2025 days averaged **86mm** of rain that day, versus just **6.9mm** on
  ordinary days — roughly **12x** more rain.
- Looking at the previous 3 days combined, suspended days averaged **241mm** versus
  **22mm** — about **11x** more.
- A handful of suspension days (4 of 19) had barely any rain *that specific day* —
  but every one of those had heavy rain in the **days just before**, and each was a
  typhoon day (or the day after one) — so these were partly suspensions called
  ahead of a storm, not just the "buildup" pattern from §8.
- A simplified, rainfall-only version of the model, taught only on 2023–2024
  and then tested on every day of 2025, correctly caught **15 of the 19** real
  suspension days, with 12 false alarms out of 346 ordinary days. The 4 it
  missed were typhoon days with little rain, when the city suspended classes
  ahead of the storm.

**Why this matters:** this is the first time in the whole study that the
rainfall–suspension relationship gets checked against a year of *completely real*
data the model was never trained on — not a held-back slice of the same dataset,
but a genuinely different year. It holding up this clearly is a meaningfully
stronger form of evidence than the cross-validation in §6, and it's a good answer
if a panelist asks "how do you know this isn't just a coincidence in your training
data?"

**What it doesn't fix:** we still don't have real temperature/humidity/wind for
2025, and the suspension log for 2025 has the same "only lists days it happened"
gap as before — so the negative-labeling assumption itself is still an assumption,
just now resting on real rainfall instead of an educated guess for at least one of
its inputs.

---

## 10. If you get asked "so is the model good or not?"

The honest, plain-language answer: **it's a reasonable proof-of-concept, not a
finished product.** It clearly does better than guessing randomly, and somewhat better than a
simple "suspend if it rained over 59mm" rule and the other programs we compared it to, and it leans on a sensible, explainable
signal (accumulated rainfall) that held up again on an independent year of real
data (§9). But with only 24 (or 19, for 2025 alone) real suspension examples to
learn from, its false-alarm rate is still fairly high, and its confidence shouldn't
be overstated. That combination — "promising, and now independently checked, but
still data-limited" — is a completely normal and defensible conclusion for an
undergraduate thesis; you don't need to claim more than that.

---

## 11. Quick glossary (for a mid-defense panic moment)

- **True positive**: model said "suspend," and it really was a suspension day.
- **False positive** (false alarm): model said "suspend," but it wasn't.
- **False negative** (missed one): model said "don't suspend," but it actually was.
- **True negative**: model said "don't suspend," and it correctly wasn't.
- **Class imbalance**: when one outcome (suspension days) is much rarer than the
  other (ordinary days) — the root cause of why accuracy alone is misleading here.
- **Feature**: one piece of input information the model looks at (e.g., rainfall,
  humidity).
- **Feature importance**: how much the model actually leaned on each feature.
- **Training set / test set**: the "study material" the model learns from, vs. the
  "exam" days it's graded on that it never saw while learning.
