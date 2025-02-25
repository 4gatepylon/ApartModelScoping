A bfief introduction of the project.

# Names of the team
Adriano, Emile, Erik, and David.

# Problem Statement and Project Overview
Right now adversarial attacks (i.e. through prompt injection, jailbreaking) are a big issue for production LLMs. We have this picture of the cat playing wackamole because AFAIK in the LLM world these attacks keep coming out and companies (such as OpenAI, Anthropic, etc...) keep having to patch them, either by refusal training on new attacks or by using text-based or embedding-based detection and deny-list methods. The reason this situation is problematic, and the reason the attacks keep coming out is that the attack surface is too large. While developers add _some_ guardrails before deployment, most safety measures are in large part reactive AFAIK. If we want to have a more reasonable level of safety before critical LLMs get pwned in production, we need to have a different approach.

For general-purpose LLMs there may no way around this (though we will briefly allude to a modification of our method that should help later), but by far in most cases LLMs are used in narrow domains. Examples include LLMs used for call centers servicing specific industries, shopping website assistants, internal company question-answering, etc. These LLMs do not need to have general-purpose capabilities, and yet because foundation models are general-purpose, because it's easiest and most performant to finetune a foundation model to your specific task instead of starting from scratch, and because the common forms of finetuning don't effectively unlearn unecessary capabilities, most LLMs that are meant to be used in a narrow domain are actually general-purpose. Your shopping website assistant LLM can code.

If we had a way to scalably unlearn unecessary capabilities from narrow LLMs, it would narrow the attack surface and space of possible impacts for a large number of usecases, thereby increasing safety systemically. Importantly, this method should not require large amounts of negative guidance, but instead a tractable amount of _positive_ guidance. We want to be able to post-train a general-purpose LLM, not to only unlearn or refuse certain actions, but also and more importantly, to unlearn and refuse EVERYTHING that is not within the scope of the tasks it is intended for. This is called "passive scoping."

Passive scoping does not necessarily need or involve new fundamental changes to the ways we train and tune LLMs. Instead, it is a paradigm shift in how we think about guardrails and safety the emphasizes the principle of least privilege, We believe it will involve many tweaks and minor modifications to existing methods so as to make guardrails more scalable.

# Presentation Overview
That is the core essence of our project. We will; explore a couple different methods that could be used to achieve this goal, our results with some very preliminary experiments.

# Architecture
Before we go into more details of the actual methods and our results, which my teammates will present, I want to give you a brief framework which describes the sets of solutions we are so far considering.

There are 3 places where passive scoping can be done:
1. **Pretraining**: we could train in a way that allows for the easy removal of capabilities, either because the model is capability-sharded (like an MoE) or because we simply exclude certain elements of the dataset. For the most part, we do not explore this in this project, since it is resource-intensive.
2. **Post-training**: we could modify and combine unlearning, refusal-training, and preference learning, pruning, finetuning, or other methods to remove capabilities from the weights of a model. Because this hackathon was so short-term, we did not do a lot of post-training, though we are curious to later try iterative pruning (and/or unlearning and/or distillation/compression/quantization) and with finetuning on in-scope tasks. We are also eager to measure the capabilities of latent adversarial training and the like.
3. **Inference-time**: this includes prompting, latent-space filtering, anomaly detection, and more. We _mainly_ rocus on inference-time methods that do not modify the weights, since they require less time and resources to tinker with. Every inference-time method we consider can be thought of like this: there is a DAG of operations (i.e. between the layers and across tokens) and at each step we optionally anomaly-detect to decide whether to break the generation chain (i.e. for refusal) and then we optionally filter (i.e. dimensionality reduction) to remove unecessary capabalities' activations. These steps can be both done in text (i.e. CoT) or activations.

# Conclusion
Now that you know why our project is important and what types of solutions we are considering, Emile, Erik, and David will:
1. Illustrate what solutions we tried out for this hackathon
2. Present some basic results on how and whether they worked
3. Elucidate in more details the benefits of passive-scoping from the point of view if these methods and connect our proposal to prior work.

# Thank you!
