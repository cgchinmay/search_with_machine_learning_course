import fasttext
import argparse

model = fasttext.load_model("/workspace/datasets/fasttext/title_model.bin")

parser = argparse.ArgumentParser(description='Process input for synonym generation')
general = parser.add_argument_group("general")
general.add_argument("--input", default="/workspace/datasets/fasttext/top_words.txt",  help="The file containing words")
general.add_argument("--threshold", default=0.75,  help="The minimum threshold for similar words")
general.add_argument("--output", default="/workspace/datasets/fasttext/synonyms.csv", help="the file to output to")

args = parser.parse_args()
input_file = args.input
threshold = args.threshold
output_file = args.output

f = open(input_file)
words = f.read().splitlines()

with open(output_file, 'w') as output:
    # for label_list in all_labels:
    for word in words:
        nn_pairs = model.get_nearest_neighbors(word)
        output.write(f'{word}')
        for (score, synonym) in nn_pairs:
            if score >= threshold:
                output.write(f',{synonym}')
        output.write('\n')
            
        



