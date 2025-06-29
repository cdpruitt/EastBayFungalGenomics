# To test this script, supply the path to a .fastq file input, e.g.
#
# python3 ValidationScript.py myInputSequence.fastq
#
# Dependencies include:
#  - biopython
#  - scikit-learn
#  - numpy
#  - regex

import os
import subprocess
from Bio import SeqIO, pairwise2
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from collections import Counter, defaultdict
import numpy as np
import regex as re
import csv
import tempfile

# ---- User Parameters ----
fastq_path = "input_sequences.fastq"
forward_primer = "CTTGGTCATTTAGAGGAAGTAA"  # ITS1F
reverse_primer = "TACTACCACCAAGATCT"  # LR7
consensus_agreement_threshold = 0.8
max_non_robust_positions = 10 # max allowed non-consensus positions 
mafft_path = "mafft"  # If MAFFT is in your PATH
minimum_amplicon_length = 500 # discard amplicons below this threshold
QC_LOG_FILE = "qc_issues.csv"

# ---- Load FASTQ sequences from file ----
def load_sequences(fastq_path):
    return list(SeqIO.parse(fastq_path, "fastq"))

# ---- Align sequences using MAFFT ----
def mafft_align(records):
    with tempfile.TemporaryDirectory() as tmpdir:
        input_fasta = os.path.join(tmpdir, "input.fasta")
        output_fasta = os.path.join(tmpdir, "aligned.fasta")

        SeqIO.write(records, input_fasta, "fasta")

        subprocess.run(
            [mafft_path, "--auto", input_fasta],
            stdout=open(output_fasta, "w"),
            stderr=subprocess.DEVNULL,
            check=True
        )

        return list(SeqIO.parse(output_fasta, "fasta"))

# ---- apply reverse-complement to a sequence ----
def reverse_complement(seq):
    return str(Seq(seq).reverse_complement())

# ---- Search for a primer sequence, allowing minor mismatches
def find_primer_fuzzy(seq, primer, max_score_drop=5):
    """
    Find primer in seq allowing mismatches and indels using local alignment.
    Returns the start index of the best alignment if found, else -1.
    """
    alignments = pairwise2.align.localms(seq, primer, 2, -1, -2, -1)  # match=2, mismatch=-1, gap open=-2, gap extend=-1

    if not alignments:
        return -1

    best = alignments[0]
    score, start, end = best[2], best[3], best[4]
    perfect_score = 2 * len(primer)

    if score >= perfect_score - max_score_drop:
        return start
    else:
        return -1

# ---- If a quality control issue occurs (primer not recognized), log this record ----
def log_qc_issue(record_id, stage, reason, fwd_idx=None, rev_idx=None):
    with open(QC_LOG_FILE, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([record_id, stage, reason, fwd_idx, rev_idx])

# ---- Reorient (reverse complement) sequences using primer key ----
def reorient_sequences_fuzzy(records, forward_primer, max_score_drop=5):
    forward_rc = reverse_complement(forward_primer)
    corrected = []
    skipped = []
    for record in records:
        seq = str(record.seq)
        fwd_idx = find_primer_fuzzy(seq, forward_primer, max_score_drop)
        rev_idx = find_primer_fuzzy(seq, forward_rc, max_score_drop)
        if fwd_idx != -1:
            corrected.append(record)
        elif rev_idx != -1:
            rc_seq = reverse_complement(seq)
            corrected.append(SeqRecord(Seq(rc_seq), id=record.id, description=""))
        else:
            skipped.append(record.id)
            log_qc_issue(record.id, "reorientation", "primer not found in either direction", fwd_idx, rev_idx)
    return corrected, skipped

# ---- Keep only the sequence region between primers ----

def trim_primers_fuzzy(records, forward_primer, reverse_rc, max_score_drop=5):
    trimmed = []
    skipped = []
    for record in records:
        seq = str(record.seq)
        fwd_idx = find_primer_fuzzy(seq, forward_primer, max_score_drop)
        rev_idx = find_primer_fuzzy(seq, reverse_rc, max_score_drop)
        if fwd_idx != -1 and rev_idx != -1 and fwd_idx < rev_idx:
            trimmed_seq = seq[fwd_idx+len(forward_primer):rev_idx]
            trimmed.append(SeqRecord(Seq(trimmed_seq), id=record.id, description=""))
        else:
            skipped.append(record.id)
            log_qc_issue(record.id, "trimming", "primer match failed or indices invalid", fwd_idx, rev_idx)
    return trimmed, skipped

# ---- Compute consensus among sequences; retains gaps ----
def compute_consensus(alignment, threshold=0.8):
    """Return consensus string and list of consensus support at each position"""
    seqs = [str(rec.seq) for rec in alignment]
    consensus = []
    support = []

    for col in zip(*seqs):
        count = Counter(col)
       #if "-" in count:
       #    del count["-"]  # ignore gaps
        if not count:
            consensus.append("N")
            support.append(0)
            continue
        base, freq = count.most_common(1)[0]
        prop = freq / sum(count.values())
        consensus.append(base if prop >= threshold else "N")
        support.append(prop)
    
    consensus_seq = "".join(consensus)
    consensus_record = SeqRecord(Seq(consensus_seq), id="consensus")
    return consensus_record, support

# ---- Extract aligned regions identified by `informative_positions` key ----
def extract_variable_sites(aligned_seqs, informative_positions):
    simplified = []
    for record in aligned_seqs:
        simplified.append("".join(record.seq[i] for i in informative_positions))
    return simplified

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score

# ---- Identify clusters of record in aligned sequences ----
def cluster_variable_sites(simplified_seqs, max_clusters=6):
    vectorizer = CountVectorizer(analyzer='char', lowercase=False)
    X = vectorizer.fit_transform(simplified_seqs).toarray()

    best_score = -1
    best_labels = None
    for k in range(2, min(max_clusters + 1, len(X))):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(X)
        if len(set(labels)) == 1:
            continue
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_labels = labels

    return best_labels

# ---- Group records according to their cluster label ----
def group_by_cluster(aligned_records, labels):
    clustered = defaultdict(list)
    for record, label in zip(aligned_records, labels):
        clustered[label].append(record)
    return clustered

# ---- Show differences between similar consensus records ----
def summarize_consensus_differences(consensus_records):
    from itertools import combinations
    from difflib import ndiff

    # Reference: consensus with the most members (or just the first)
    reference = consensus_records[0]
    ref_seq = str(reference.seq)
    print(f"Using cluster 0 as reference\n")

    for idx, record in enumerate(consensus_records[1:], start=1):
        comp_seq = str(record.seq)
        print(f"--- Differences from cluster 0 vs cluster {idx} ---")
        diffs = [(i, ref_base, comp_base)
                 for i, (ref_base, comp_base) in enumerate(zip(ref_seq, comp_seq))
                 if ref_base != comp_base]
        for i, ref_b, comp_b in diffs:
            print(f"  Pos {i}: {ref_b} → {comp_b}")
        print(f"Total differences: {len(diffs)}\n")

    # Pairwise % divergence
    print("Pairwise divergence matrix (in % differences):")
    n = len(consensus_records)
    matrix = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        s1, s2 = str(consensus_records[i].seq), str(consensus_records[j].seq)
        length = min(len(s1), len(s2))
        mismatches = sum(1 for a, b in zip(s1[:length], s2[:length]) if a != b)
        pct_diff = 100 * mismatches / length
        matrix[i, j] = matrix[j, i] = pct_diff

    for row in matrix:
        print("  " + "  ".join(f"{v:5.1f}" for v in row))

# ---- Eliminate any gap-dominated columns from an alignment (due to spurious insertions in a few records) ----
def filter_gap_dominated(aligned_records, gap_threshold=0.8):
    alignment_length = len(aligned_records[0].seq)
    num_seqs = len(aligned_records)
    keep_positions = []

    for i in range(alignment_length):
        gap_count = sum(1 for rec in aligned_records if rec.seq[i] == '-')
        if gap_count / num_seqs <= gap_threshold:
            keep_positions.append(i)

    filtered = []
    for record in aligned_records:
        new_seq = ''.join(record.seq[i] for i in keep_positions)
        filtered.append(SeqRecord(Seq(new_seq), id=record.id, description=""))
    return filtered

# --- main workflow ----
def main(fastq_path):
    # Write QC log header
    with open(QC_LOG_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["read_id", "stage", "reason", "fwd_idx", "rev_idx"])

    print("Loading sequences...")
    sequences = load_sequences(fastq_path)

    print("Reorienting sequences with reverse complement of forward primer...")
    reoriented, skipped_orientation = reorient_sequences_fuzzy(sequences, forward_primer)

    print("Trimming primer regions...")
    reverse_rc = reverse_complement(reverse_primer)
    trimmed, skipped_trimming = trim_primers_fuzzy(reoriented, forward_primer, reverse_rc)

    print(f"Skipped during orientation: {skipped_orientation}")
    print(f"Skipped during trimming: {skipped_trimming}")

    print("Aligning sequences with MAFFT...")
    aligned = mafft_align(trimmed)

    print("Filtering gaps...")
    aligned_filtered = filter_gap_dominated(aligned)

    print("Computing consensus sequence...")
    consensus_record, support = compute_consensus(aligned_filtered, threshold=0.8)
    SeqIO.write(aligned_filtered, "aligned_reads.fasta", "fasta")
    SeqIO.write(consensus_record, "consensus.fasta", "fasta")

    # test for sub-consensus clusters within the alignment
    informative_positions = [i for i, s in enumerate(support) if s < 0.8]
    if not len(informative_positions):
        # alignment in complete agreement -- we're done
        return

    # alignment not in complete agreement: characterize clusters
    simplified = extract_variable_sites(aligned_filtered, informative_positions)
    labels = cluster_variable_sites(simplified)
    clustered = group_by_cluster(aligned_filtered, labels)

    consensus_records = []
    for cluster_id, cluster_reads in clustered.items():
        filename = f"cluster_{cluster_id}_alignment.fasta"
        SeqIO.write(cluster_reads, filename, "fasta")
        sub_consensus, _ = compute_consensus(cluster_reads, threshold=0.8)
        SeqIO.write(sub_consensus, f"cluster_{cluster_id}_consensus.fasta", "fasta")

        consensus_records.append(sub_consensus)
        
    summarize_consensus_differences(consensus_records)

if __name__ == "__main__":
    import sys
    main(sys.argv[1])
