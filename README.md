# ApartModelScoping
Model-scoping project from Apart sprint in 2024-01-17 (https://www.apartresearch.com/event/ai-safety-hackathon)

# Repository Structure
The way this repository is structured is pretty basic. We make a new folder for each "experiment". An experiment basically is meant to be a unit of information acquisition or demonstration of capability or lack thereof. For the most part, experiments for us will test capabilities of different methods to maintain models within scopes that we define. **Each experiment has a single _owner_**. Multiple people are welcome to contribute to one experiment but it's on the owner to manage this. We will discuss periodically about what experiments we need to run, assign them, and then each person will PR a new experiment in a folder. We will then synthesize everything in the end into a 4-page writeup.

Each experiment must have:
1. A `README.md` that (1) says who the owner is, (2) describes the experiment and obviously any relations to others, what needs to get done, etc...
2. Your code, requirements if needed (by default you can use `requirements.txt`), etc...

I recommend using conda with `conda create -n <env_name> python=3.12` and then `conda activate <env_name>`. You can then install requirements with `pip install -r requirements.txt` and you can use a `.env` file for any private tokens, etc... and use `export $(cat .env | xargs)` to load them. Anything simple should be fine.

# Point of the Project
**The goal of the project is to be able to limit a model to only behave in a specific way, robustly, _without needing to define what to avoid_.** Normally, this will correspond, for us, to answering question of a specific topic only and not others (i.e. answering only questions about online products). IMO (adriano), In general, it is OK if the model retains capabilities that are not related to the scope, but we usually will seek to make sure that a specific capability is removed. **They key difference with unlearning and finetuning/rlhf is that we need to remove _unknown_ capabilities.** In practice that means that the method we use for removal should not assume any knowledge of what it is removing. This is a good safety capability to have to make finetuning for safety easier (there is less to cover and therefore a lower chance of missing something) and it can defend against trojans, of which there are not great defenses yet AFAIK (adriano).

Generally the classes of approaches we are looking into are:
1. Training-based approaches (i.e. untargetted LAT)
2. Prompt and inference-time compute-based approaches (i.e. "rewrite this sentence" and re-prompt)
3. Latent space filtering-based approaches (i.e. PCA, SAEs of which there are various ways of using them)
4. Latent space anomaly detection-based approaches (i.e. logistic regression)
5. Pruning, model compression, or other such ideas... (haven't looked too much into this so far---adriano)

# What TODO
If you are new to this repo something you can do is basically find an unclaimed (ownerless) experiment and then claim it and implement it. Discuss with adriano for more questions.