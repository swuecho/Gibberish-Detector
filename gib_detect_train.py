#!/usr/bin/python

import math
import pickle
import json
import gib_detect

accepted_chars = 'abcdefghijklmnopqrstuvwxyz '

pos = {char: idx for idx, char in enumerate(accepted_chars)}


def train():
    """ Write a simple model as a pickle file """
    k = len(accepted_chars)
    # Assume we have seen 10 of each character pair.  This acts as a kind of
    # prior or smoothing factor.  This way, if we see a character transition
    # live that we've never observed in the past, we won't assume the entire
    # string has 0 probability.
    counts = [[10 for i in range(k)] for i in range(k)]

    # Count transitions from big text file, taken
    # from http://norvig.com/spell-correct.html
    for line in open('big.txt'):
        for a, b in gib_detect.ngram(2, line):
            counts[pos[a]][pos[b]] += 1

    # Normalize the counts so that they become log probabilities.
    # We use log probabilities rather than straight probabilities to avoid
    # numeric underflow issues with long texts.
    # This contains a justification:
    # http://squarecog.wordpress.com/2009/01/10/dealing-with-underflow-in-joint-probability-calculations/
    for i, row in enumerate(counts):
        s = float(sum(row))
        for j in range(len(row)):
            row[j] = math.log(row[j] / s)

    # Find the probability of generating a few arbitrarily choosen good and
    # bad phrases.
    good_probs = [gib_detect.avg_transition_prob(
        l, counts) for l in open('good.txt')]
    bad_probs = [gib_detect.avg_transition_prob(
        l, counts) for l in open('bad.txt')]

    # Assert that we actually are capable of detecting the junk.
    assert min(good_probs) > max(bad_probs)

    # And pick a threshold halfway between the worst good and best bad inputs.
    thresh = (min(good_probs) + max(bad_probs)) / 2
    model = {'mat': counts, 'thresh': thresh}
    pickle.dump(model, open('gib_model.pki', 'wb'))
    with open('gib_model.json', 'w') as outfile:
        json.dump(model, outfile)


if __name__ == '__main__':
    train()
