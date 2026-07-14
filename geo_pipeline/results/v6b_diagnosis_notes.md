# v6b Full Diagnosis Notes

Source result: `geo_pipeline/results/v6b_full.json` on 4536 YFCC4K records.

## Metrics

```text
Street <1km        5.51%
City <25km        16.45%
Region <200km     26.54%
Country <750km    44.14%
Continent <2500km 62.74%
```

## Diagnosis

The main bottleneck is not geocoding. `missing` geocode count was only 4 and
country-child conflict rate was 3.57%.

The dominant error is country-level North America bias, especially United States:

```text
North America false positives: n=825
  by GT continent: Africa:33, Asia:102, Europe:575, Oceania:52, South America:63
  by predicted country: united states:735, canada:59, mexico:23, costa rica:8
```

Predicted Europe is reliable (`pred Europe acc@2500=87.34%`), while predicted
North America is weak (`pred North America acc@2500=49.48%`). Many European
images are being over-routed to United States/Canada.

## v7 Changes To Evaluate

- Require at least one country/city verification step before POMDP threshold stop.
- Add prompt guard against defaulting to United States/Canada from English text,
  generic roads, vegetation, architecture, media, or product branding alone.
- Pass top-3 country candidates into city reasoning and top country/city
  candidates into street reasoning instead of only the MAP parent.
- Parse and merge wrapper responses shaped like `{general, cue_responses}`.
- In evaluation geocoding, if city/street already embeds a country that conflicts
  with the parent country, geocode the embedded-country child first.

## Compare Next Result Against v6b

When `v7_full.json` is available, compare:

```text
Continent <2500km
Country <750km
North America false positives
pred=North America row in predicted_vs_gt_continent
pred Europe / pred North America acc@2500
Country-child conflict rate
geocode_source
```

Primary target: `North America false positives` should drop meaningfully below
825 without hurting Europe accuracy.

## v7 Full Result And Regression

`v7_full.json` reduced North America false positives sharply, but over-corrected
and hurt overall accuracy:

```text
Total records: 4516
Street <1km        3.57%
City <25km        12.98%
Region <200km     21.30%
Country <750km    37.93%
Continent <2500km 56.71%
North America false positives: n=120
Country-child conflict rate: 51.84%
```

The key regression is not North America bias anymore. It is child-level country
override: `child_country_override: 1877`, `street_embedded_country: 1871`, and a
51.84% country-child conflict rate. The v7 prompt encouraged child levels to
introduce alternative countries, and evaluation trusted embedded child countries
over the parent country. This often geocoded an incorrect child country even
when the country-level prediction was usable.

## v8 Fix Direction

- Keep wrapper parsing and the "at least one verification step before stop"
  rule.
- Remove evaluation-time child-country override.
- Skip conflicting child country geocodes and fall back to lower-risk city or
  country-level geocoding instead of querying mixed-country strings.
- Tighten city/street context so child levels refine within top country/city
  candidates and do not introduce countries outside those candidates unless
  directly visible in the image.

For v8, compare against both v6b and v7. Desired pattern: North America false
positives stay far below 825, while country-child conflict returns near the v6b
range instead of 51.84%, and continent/country accuracy recover above v7.

## Forward Plan After v8 Finishes

Wait for the current v8 run before changing inference again. First decide if v8
fixed the v7 regression:

```text
Country-child conflict rate: should drop far below 51.84%
Continent <2500km: should recover above v7's 56.71%
North America false positives: should remain far below v6b's 825
```

If v8 is directionally good, add the next three GeoBayes-style state-control
mechanisms, preferably one stage at a time and validate with `--limit 300`:

1. Entropy / margin stop: do not descend just because top-1 is slightly highest;
   require low uncertainty or sufficient top1-top2 margin.
2. Replace: if the current candidate set is flat, unsupported, or exhausted
   without a stable posterior, regenerate candidates instead of forcing a weak
   MAP answer.
3. Backtrack: if city/street evidence strongly conflicts with the parent country,
   route that evidence back to the country level and re-evaluate rather than
   geocoding a cross-country child prediction.

After those are stable, try the higher-innovation directions:

A. POMDP-inspired evidence selection with explicit utility/EIG-lite: score actions
   by expected entropy reduction or margin improvement minus cost, instead of
   relying only on the LLM policy choice.
B. Multi-signal transition controller: combine top posterior, margin, entropy,
   DST conflict, SL uncertainty, country-child conflict, and evidence exhaustion
   for stop/replace/backtrack decisions.
C. Continent guard / continent posterior regularization: aggregate country
   posterior to continent posterior; if the continent is unstable or child
   predictions cross continents, block descent or trigger backtrack.

Positioning for writing: this should be framed as extending GeoBayes with
POMDP-inspired evidence selection and multi-signal hierarchical control, not as
a full exact POMDP solver. Current POMDP module is action selection over the
posterior belief state; it does not yet model `O(o|s,a)` or solve exact EIG.

## v8 Full Result And Diagnosis

`v8_full.json` did not recover from v7. It removed evaluation-time child country
override, but the model/pipeline still produced many cross-level conflicts and
over-corrected away from North America:

```text
Street <1km        3.75%
City <25km        12.43%
Region <200km     20.17%
Country <750km    37.32%
Continent <2500km 57.54%
Unknown country rate: 0.11%
Country posterior top mass: mean=0.5188 median=0.5064
Country-child conflict rate: 42.90%
```

Key continent confusion:

```text
GT North America acc@2500 = 35.94%
pred North America n=893, acc@2500=70.55%
GT North America misrouted to Europe=519, Asia=333, Oceania=95
```

Interpretation: v6b over-predicted North America (`pred NA n=2381`, many false
positives). v7/v8 over-corrected and now under-predict North America (`pred NA
n=893` while GT NA n=1775). The top country posterior is too flat (~0.52), and
city/street still frequently propose locations inconsistent with the parent
country. Evaluation no longer lets child country override parent, but the model
output itself is still inconsistent, so the pipeline often falls back to bare
street/city or country-level geocoding.

Next code direction should not further strengthen anti-US/Canada prompting.
Instead:

1. Add entropy/margin gating before descending: if country posterior is flat,
   do not proceed to city/street; replace or keep collecting country evidence.
2. Add Replace at country level: regenerate candidates when top mass and margin
   remain weak after verification.
3. Add child hypothesis filtering/backtrack: if city/street hypotheses embed a
   country outside the current country candidate set, treat them as conflict
   evidence, not as normal child candidates.
4. Calibrate the country prompt to be anti-default, not anti-North-America:
   avoid defaulting to US/Canada from weak cues, but explicitly allow North
   America when traffic, signage, road design, or landmarks support it.
5. Then add continent guard: aggregate country posterior to continents and block
   descent when continent posterior is unstable or flips across levels.

## v9 Implemented State-Control Changes

Implemented after v8 regression:

- Entropy / margin gate: `_stable_for_descent()` now requires either strong top
  posterior (`STRONG_POSTERIOR_THR=0.70`) or `TRANSITION_THR` plus sufficient
  top1-top2 margin (`STABLE_MARGIN_THR=0.10`) or low normalized entropy
  (`STABLE_ENTROPY_THR=0.85`). Flat country posteriors no longer blindly drive
  city/street prompts.
- Country Replace: if country posterior remains unstable after verification,
  the pipeline runs one replacement hypothesize+verify pass with a prompt asking
  for diverse, evidence-grounded country candidates. The replacement prompt is
  anti-default, not anti-North-America.
- Backtrack/filter guard: city/street hypotheses that embed countries outside
  the top country candidates are removed from the child posterior. If all child
  candidates conflict, the child level becomes `Unknown` and the pipeline falls
  back to the parent level rather than geocoding cross-country children.
- Prompt calibration: the country prompt now says not to default to any country
  from weak generic cues, while explicitly allowing North America when concrete
  road signs, traffic infrastructure, license plates, landmarks, vegetation, or
  architecture support it.
- Diagnostics: `evaluate.py` records `country_stable`, `country_replaced`, and
  child backtrack conflicts; `analyze_results.py` prints replace/stable/backtrack
  rates.

Next run should be `v9_limit300` first. Watch:

```text
Country replace rate
Country stable rate
Backtrack conflict rate
Country-child conflict rate
GT North America acc@2500
Continent <2500km
```
