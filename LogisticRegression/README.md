Owner: 4gate (https://github.com/4gatepylon)
Date last updated: Noon 2024-01-19

The idea of this experiment is pretty straightforward: we are going to just perform some logistic regression on our models to see if we can classify based on task, especially if we are out of the "desired" task. If this works then we are in a strong position. We try the following versions from most simple/easy to least:
1. Simply preform logistic regression on the hidden states (i.e. regular linear probing). We are scanning layers and using a small dataset that we choose a random split for and do cross-validation on (we might do 10-fold or 5-fold depending on whether our dataset is reasonably-sized or very small)
2. (optional) Linear probing on multiple layers or multiple tokens at once (we might fold this into (1)).
    - Start with last token only; we can then try average. TODO(adriano) not sure how to deal with the lack of knowledge of a priviledged basis. Maybe that's where the SAEs come in? NOTE: the previous paper used the following strategy which we should reproduce: "mean aggregate our SAE feature matrices across sequence position and then zero-center and rescale our activations to be unit variance" (or maybe it need not be like this for non-SAE features?)
    - NOTE: use stratified splitting!
    - Start by training everything as a _binary_ classification problem and we might extend from there to a multi-class (binary, because we are only interested in the presence or lack thereof of malicious prompts as well as anomalies as well as on-task-ness, in general).
3. (optional) Non-linear probing akin to (1).
4. (optional) Linear probing on SAE latent-space (initially with SAEs that were trained only on biology, later with SAEs that were trained on tons of stuff; this second one will require more training). We may also try non-linear probing on SAE latent-space. 

The optional ones only if we are not able to succeed with the previous ones.

NOTE: this is an extension of: https://www.apartresearch.com/project/classification-on-latent-feature-activation-for-detecting-adversarial-prompt-vulnerabilities.

We use spylab always and we are probing across a set of datasets for three main tasks:
- Probing the inputs to the model (i.e. last token or a series of tokens from the prompt)
- Probing the outputs of the model
- Probing the entire completion request and response (so the concatenation of the two above)

The classes we mainly care about, which may overlap are:
- `malicious`
- `has_trojan`
- `biology` (our hypothetical "want to be on task" example; camel.ai)
- `math` (tbd)
- `physics` (camel.ai)
- `leetcode` (tbd)
- `shopping` (tbd)

Each one we will train as a BINARY classification problem.