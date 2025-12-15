#!/usr/bin/env python3

COMP = {'A':'T', 'T':'A', 'C':'G', 'G': 'C', 'N':'N'}
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(['A', 'T', 'G', 'C', 'N'])

def tokenize_fasta(path, k_size): # Goal is to return {header: [embeddings]}
    d = {}
    with open(path) as file:
        header = ''
        sequence = ''
        is_header = 0
        for line in file:
            l = line.strip()
            if l[0] == '>':
                header = l
                is_header = 0
            else:
                is_header = 1

            if is_header == 1:
                sequence += l
        d[header] = stream_kmers(sequence, k_size)
    return d


def stream_kmers(sequence, k_size): # doesn't stream k-mers yet, right now it saves k-mers in a list and returns the list
    k_mers_list = []
    for index in range(0, len(sequence)-k_size+1):
        k_mers_list.append(k_mer2embed(canonical(sequence[index:index+k_size])))

    return k_mers_list

def k_mer2embed(k_mer): # return embedding for a single k-mer
    label_encodings = le.transform(list(k-mer))
    ### TODO! return an embedding of a k-mer from a pre-trained nucleotide transformer
    return label_encodings

def canonical(sequnece): # take a sequence, and return its caonical form
    revc = revcomp(sequence)
    if revc > sequence:
        return sequence
    return revc

def revcomp(nucl): # take a nucleotide and return its reverse complement
    revc = ''
    for i in nucl[::-1]:
        revc += COMP[i]
    return revc
