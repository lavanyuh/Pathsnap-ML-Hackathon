"""Hidden Markov Model decoding for GPS-to-road map matching."""

import numpy as np


def viterbi_decode(emission_probs, transition_probs):
    """
    Decode the most likely sequence of road candidates.

    emission_probs: (time_steps, candidates) probabilities
    transition_probs: (time_steps - 1, candidates, candidates)
    Returns: best candidate index at each time step.
    """
    emissions = np.asarray(emission_probs, dtype=float)
    transitions = np.asarray(transition_probs, dtype=float)

    if emissions.ndim != 2:
        raise ValueError("emission_probs must be a 2D array")

    steps, states = emissions.shape

    if steps == 0 or states == 0:
        return np.array([], dtype=int)

    if transitions.shape != (max(steps - 1, 0), states, states):
        raise ValueError("Invalid transition_probs shape")

    eps = 1e-12
    log_emission = np.log(np.clip(emissions, eps, 1.0))
    log_transition = np.log(np.clip(transitions, eps, 1.0))

    scores = np.zeros((steps, states))
    backpointers = np.zeros((steps, states), dtype=int)

    scores[0] = log_emission[0]

    for t in range(1, steps):
        candidates = (
            scores[t - 1][:, None]
            + log_transition[t - 1]
        )

        backpointers[t] = np.argmax(candidates, axis=0)

        scores[t] = (
            np.max(candidates, axis=0)
            + log_emission[t]
        )

    path = np.zeros(steps, dtype=int)
    path[-1] = np.argmax(scores[-1])

    for t in range(steps - 2, -1, -1):
        path[t] = backpointers[t + 1, path[t + 1]]

    return path


def match_candidates(emission_probs, candidate_arc_ids, transition_probs):
    """
    Return the most likely road arc ID for every GPS observation.
    """
    candidate_arc_ids = np.asarray(candidate_arc_ids)

    if candidate_arc_ids.shape != np.asarray(emission_probs).shape:
        raise ValueError("Candidate IDs and emissions must have equal shapes")

    path = viterbi_decode(emission_probs, transition_probs)

    if len(path) == 0:
        return np.array([], dtype=int)

    return candidate_arc_ids[np.arange(len(path)), path]
def viterbi_decode_variable(emissions, transitions):
    """
    Viterbi decoder supporting a different number of candidates
    at each GPS observation.

    emissions: list of probability arrays, one per GPS point.
    transitions: list of matrices between consecutive GPS points.
    Returns: best candidate index at each point.
    """
    if len(emissions) == 0:
        return np.array([], dtype=int)

    if len(transitions) != len(emissions) - 1:
        raise ValueError("Invalid number of transition matrices")

    log_emissions = [
        np.log(np.clip(np.asarray(e, dtype=float), 1e-12, 1.0))
        for e in emissions
    ]

    scores = log_emissions[0].copy()
    backpointers = []

    for t in range(1, len(emissions)):
        trans = np.asarray(transitions[t - 1], dtype=float)

        if trans.shape != (len(emissions[t - 1]), len(emissions[t])):
            raise ValueError("Invalid transition matrix shape")

        log_trans = np.log(np.clip(trans, 1e-12, 1.0))

        candidates = scores[:, None] + log_trans
        backpointers.append(np.argmax(candidates, axis=0))

        scores = np.max(candidates, axis=0) + log_emissions[t]

    path = np.zeros(len(emissions), dtype=int)
    path[-1] = np.argmax(scores)

    for t in range(len(emissions) - 1, 0, -1):
        path[t - 1] = backpointers[t - 1][path[t]]

    return path
