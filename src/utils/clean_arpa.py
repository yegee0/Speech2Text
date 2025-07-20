dirty_arpa = "DLA/hw_1/ASR-DeepSpeech2/3-gram.pruned.1e-7.arpa"
clean_arpa = "DLA/hw_1/ASR-DeepSpeech2/clean_3-gram.pruned.1e-7.arpa"

with open(dirty_arpa, 'r') as f1:
    with open(clean_arpa, "w") as f2:
        for line in f1:
            f2.write(line.lower())