Last updated: 2024-01-19 at 9:00 PM EST.

Current owner: TODO (no one).

Current collaborators: TODO (no one).

# TODO
Currently, I (adriano) have been using the following repository to get trojanned models: https://github.com/ethz-spylab/rlhf_trojan_competition (aka Spylab repository). I also have some that I trained that are based on the post-training code from: https://github.com/thestephencasper/latent_adversarial_trainin. I will share the weights for this second one at some point soon and it will appear here: TODO(Adriano).

If you read the `README.md` in root you know what the overarching goal is. I started a basic jupyter notebook to play around with prompt-based baselines but I didn't actually get very far so I'm leaving this experiment to get picked up and finished. It has some other stuff regarding PeFT but the main idea here is to finish and improve the section that is based on the class:
```python
class Generator:
    """
    Class you should subclass for generators (NOTE that some generators involve 
    inference-time compute).
    """
    def __init__(self, **kwargs):
        raise NotImplementedError("Subclass must implement this method")
    def generate(self, input: str, **kwargs) -> str:
        raise NotImplementedError("Subclass must implement this method")
    def get_logprob(self, question: str, response: str, **kwargs) -> float:
        raise NotImplementedError("Subclass must implement this method")
```

How to implement this interface, etc... is up to you but the high level idea is that it _may_ be possible to get a baseline for model scoping by using prompting. Specifically, one pretty bad baseline to initially start out with is to just tell the model someting like `"only anwer questions about <topic>"`. Because are using models from the Spylab repository they have backdoors and so this should not work. However, it's unclear if some other approaches _might_ work. Specifically, I want to get a strongish baseline using a stategy where we use an LLM to reword the prompt to then pass back into itself. You could also explore looking at the response from the LLM and asking itself whether it answers the question, or placing the prompt in a template and asking if it's on-topic or not before answering.

### The key deliverables here are
- [ ] Identify a finite set of the strong baselines to use (preliminary ones above)
- [ ] Define a scope and model to experment with. I recommend using the first spylab model to begin with and all of them later when you get something working. I recommend using a biology scope. I had some reasonable results with Camel.ai's biology dataset (i.e. ctrl/command+f to look for `biology_dataset = load_dataset("camel-ai/biology",`)
- [ ] You have some sample malicious prompts I made based on existing benchmarks. You can use those or whatever is best to verify that the spylab trojans are indeed going through (and you can get used to the proper preprocessing/prompt template for these models).
- [ ] (re)Verify that saying `"only answer questions about <topic>"` does not work, measuring the failure rate.
- [ ] Measure the failure rate for the stronger prompting baselines you identified.
- [ ] (recommended for the two above) Setup an LLM as judge using OpenAI or Anthropic API (recommended to use LiteLLM, look into `llm_as_judge.py` file) to be able to evaluate whether or not trojans are going through. Make sure to measure how much you agree with the LLMAsJudge prompt. You could also do this manually, but if you decide to scale to more than 100 examples, it will be very painful. It should cost no more than $1-$10 to do all the LLMAsJudge necessary for the project.
- [ ] (recommended) Clean up and improve the interface for a `Generator`. Specifically, it should be possible to slot in as a text model, but also be able to abstract away CoT-style defenses as are proposed above.
- [ ] Report the defense rates of the different baselines you tried
- [ ] Measure and report the competence of the different solutions you found (I'm thinking something manual for very small cases, but otherwise you can try with the LLMAsJudge; might depend on biology knowledge, though; also, I'm thinking mainly coherence and the boolean of whether the model actually answered the question; another thing you could try is to look at the logprobs of the responses). 
- [ ] (recommended) Generate a pareto curve of competence vs. defense rate/trojan removal power. There are two versions I have in mind: (1) the competence rate or some harmonic average of the relevant ones vs. defense rate; (2) the log-probs of the desired responses vs. undesired responses (where maybe an undesired response to a malicious prompt is "Yea sure" and you can normalize the log-probs based on length of the responses... you could also just look at the first token for Yea vs. Sorry or something like that---make sure you get a notion of what the refusal vs. compliance first-tokens are if you do this, since they are NOT yes/no and in my experiernce usually correspond to things like: Yea, Sure, Sorry, I, As, ...)

For this, you will need a GPU of some form to host the model.

An example runthrough would be something like: you get a GPU from lambda labs and use an initial test-set of 10 questions (manual grading). You use the biology dataset and ask the model to only answwer questions about biology: it doesn't work. Then you try getting the model to re-word the prompt and it works OK. You plot a pareto curve the log-probs and a pareto-curve of the rates. You find cases where the model fails by modifying the attack and then improve the CoT 1 or 2 times. If it seems really hard to use prompting, this is not bad at all, since it means that activation-based approaches which I have explored and will be working on (adriano) have merit. If prompt-based approaches "just work" then activation-based approaches might not be so important yet.