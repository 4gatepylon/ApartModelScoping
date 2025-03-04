"""
A set of modules to perform regression (which for us, for now, is primarily just _logistic_ regression) on the topic
used in our models. The key things that need to be done are:
1. Load the relevant model and store, asynchronously, the dataset of hidden states.
XXX

TODO(adriano): implement regression on more relevant latent-space dimensions SAEs (has as bearing, primarily, on the normalization).
"""

# XXX we'll implement this after we finish the logistic regression tasks ipynb to a point that was satisfactory for the hackathon.
# XXX todo tasks right now
# 1. Get a real diverse set of datasets/questions to train the regression on. Ideally we can include 10-20 topics and around 1000-10_000 questions per topic.
# 2. Enable caching for the hidden-states. Understand how much space it will take/what we store; this should also have a pydantic schema that defines the details
#    of the structure we are using. A back of the envelope calc. suggests we take 200+TB to store _everything_ (20 * 20000 * 4096 * 33 * 512 * 8 / 10^12) so we cannot do
#    that (though it may be interesting to look into compression later). THE GOAL HERE IS TO BE ABLE TO RUN THE NOTEBOOK AGAIN AND AGAIN AND HAVE IT BE REAL-TIME.
# 3. Try different forms of regression.
# 4. Make sure this is SAE slot-in-able (or it's also possible to slot in different latent dimensions, etc...)