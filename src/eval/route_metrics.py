import numpy as np
from collections import Counter


def route_continuity(network, pred_arcs):
    """
    Measure how well consecutive predicted road segments connect.

    A = starting node of each road.
    B = ending node of each road.

    Returns:
        continuity: Fraction of connected transitions.
        breaks: Number of disconnected transitions.
    """

    # Convert predictions into a NumPy array
    pred_arcs = np.asarray(pred_arcs, dtype=int).ravel()

    # Get starting and ending nodes
    A = np.asarray(network.A)
    B = np.asarray(network.B)

    # Remove consecutive repeated arc IDs
    collapsed = []

    for arc in pred_arcs:
        if not collapsed or arc != collapsed[-1]:
            collapsed.append(arc)

    # Initialize counters
    connected = 0
    breaks = 0
    total = 0

    previous_arc = None

    for arc in collapsed:

        # Treat -1 as a gap
        if arc == -1:
            previous_arc = None
            continue

        # Check whether arc ID is valid
        if arc < 0 or arc >= len(A):
            previous_arc = None
            continue

        # Compare consecutive roads
        if previous_arc is not None:

            total += 1

            if B[previous_arc] == A[arc]:
                connected += 1
            else:
                breaks += 1

        previous_arc = arc

    # Calculate continuity
    continuity = connected / total if total > 0 else 1.0

    return continuity, breaks


def sequence_metrics(network, pred_arcs, reference_route):
    """
    Calculate length-weighted precision, recall and F1-score.

    Repeated arc traversals are counted separately.
    Invalid predicted arc IDs are excluded.
    """

    # Convert routes into lists
    predicted = np.asarray(pred_arcs, dtype=int).ravel().tolist()
    reference = np.asarray(reference_route, dtype=int).ravel().tolist()

    # Get road lengths
    lengths = np.asarray(network.length, dtype=float)

    # Count occurrences of each valid predicted road
    pred_counts = Counter(
        arc for arc in predicted if 0 <= arc < len(lengths)
    )

    # Count occurrences of each valid reference road
    ref_counts = Counter(
        arc for arc in reference if 0 <= arc < len(lengths)
    )

    # Calculate total predicted road length
    predicted_length = sum(
        count * lengths[arc]
        for arc, count in pred_counts.items()
    )

    # Calculate total reference road length
    reference_length = sum(
        count * lengths[arc]
        for arc, count in ref_counts.items()
    )

    # Find correctly matched road traversals
    common = pred_counts & ref_counts

    correct_length = sum(
        count * lengths[arc]
        for arc, count in common.items()
    )

    # Calculate precision
    precision = (
        correct_length / predicted_length
        if predicted_length > 0 else 0.0
    )

    # Calculate recall
    recall = (
        correct_length / reference_length
        if reference_length > 0 else 0.0
    )

    # Calculate F1-score
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0.0
    )

    return precision, recall, f1
