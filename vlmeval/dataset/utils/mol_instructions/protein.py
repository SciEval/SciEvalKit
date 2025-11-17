# molecule task
# https://github.com/zjunlp/Mol-Instructions/tree/main/evaluation/molecule

import json
import os
import re
import numpy as np
from os import environ
from typing import List, Optional

from Bio.Seq import Seq

from Bio.Align import PairwiseAligner, substitution_matrices
import math


def normalized_smith_waterman(seq1, seq2, matrix_name="BLOSUM45", open_gap=-10, extend_gap=-0.5):
    """
    Compute normalized Smith-Waterman score for protein sequences.

    Args:
        seq1, seq2 (str): Protein sequences (uppercase letters)
        matrix_name (str): Name of substitution matrix (default: BLOSUM62)
        open_gap (float): Gap opening penalty
        extend_gap (float): Gap extension penalty

    Returns:
        float: Normalized score between 0.0 and 1.0
    """
    # Initialize aligner
    aligner = PairwiseAligner()
    aligner.mode = 'local'  # Smith-Waterman algorithm
    aligner.open_gap_score = open_gap
    aligner.extend_gap_score = extend_gap

    # Load substitution matrix
    try:
        matrix = substitution_matrices.load(matrix_name)
    except ValueError:
        raise ValueError(f"Matrix {matrix_name} not available. Try: {substitution_matrices.load()}")

    # Set substitution matrix
    aligner.substitution_matrix = matrix

    # Calculate raw alignment score
    raw_score = aligner.score(seq1, seq2)
    if raw_score <= 0:
        return 0.0

    # Calculate self-alignment scores
    def calc_self_score(seq):
        """Calculate maximum possible self-alignment score"""
        score = 0
        for aa in seq:
            try:
                # Try direct lookup
                score += matrix[aa, aa]
            except KeyError:
                # Try reverse lookup for symmetric matrices
                score += matrix[aa, aa]  # Same residue
        return score

    self_score1 = calc_self_score(seq1)
    self_score2 = calc_self_score(seq2)

    # Handle invalid self-scores
    if self_score1 <= 0 or self_score2 <= 0:
        return 0.0

    # Compute normalization factor (geometric mean)
    norm_factor = math.sqrt(self_score1 * self_score2)

    return min(raw_score / norm_factor, 1.0)


