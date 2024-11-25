import argparse
from tqdm import tqdm

from tokenizer.character_tokenizer import CharacterTokenizer

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
# 
#     parser.add_argument("--csv", type=str, required=True)
# 
#     args = parser.parse_args()
# 
#     translation_csv = args.csv
# 
#     lines = []
#     with open(translation_csv, "r") as file:
#         file.readline()
#         for line in tqdm(file):
#             lines.append(line.strip().split("\t"))
# 
#     # save smaller csv
# 
#     with open("smaller.csv", "w") as file:
#         for line in lines[:100000]:
#             file.write("\t".join(line) + "\n")
#         file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", type=str, required=True)

    args = parser.parse_args()

    translation_csv = args.csv

    tokenizer = CharacterTokenizer()
    lines = []
    print("hi")
    with open(translation_csv, "r") as file:
        file.readline()
        for line in tqdm(file):
            split_line = line.strip().split(",")
            french_sentence, english_sentence = split_line[0], split_line[1]
            french_encoded = tokenizer.encode(french_sentence)
            english_encoded = tokenizer.encode(english_sentence)

    print("final character set:", tokenizer.characters)
