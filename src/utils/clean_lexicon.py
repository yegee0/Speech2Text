# CLEANING LEXICON FILES FROM ' SIGN

dirty_lexicon = open("DLA/hw_1/ASR-DeepSpeech2/lexicon.txt", "r")
clean_lexicon = open("DLA/hw_1/ASR-DeepSpeech2/clean_lexicon.txt", "w+")

while line := dirty_lexicon.readline():
    line = line.lower().replace("'", "")
    print(line, end="", file=clean_lexicon)

dirty_lexicon.close()
clean_lexicon.close()