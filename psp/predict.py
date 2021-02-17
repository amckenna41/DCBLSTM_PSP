# #Call to predict with model
# import fastaparser
#
# #get rid of local_main func
#
# def main(args):
#     pass
#
# def parse_sequence(args):
#
# def parse_fasta():
#
# >>> with open("fasta_file.fasta") as fasta_file:
#         parser = fastaparser.Reader(fasta_file)
#         for seq in parser:
#             # seq is a FastaSequence object
#             print('ID:', seq.id)
#             print('Description:', seq.description)
#             print('Sequence:', seq.sequence_as_string())
#             print()
#
#     #parse fasta
#     #if input not of type fasta then return
#     pass
#
# def predict():
#     pass
#
#
# if __name__ == "__main__":
#
#     #initialise input arguments
#     parser = argparse.ArgumentParser(description='Predicting from model')
#
#     parser.add_argument('-model', '--model', required = True,
#                     help='Path of model to predict')
#     parser.add_argument('-seq', '--sequence', required = True, default ="all",
#                     help='Protein sequence to predict secondary structure')
#     args = parser.parse_args()
#
#     main(args)
#
#
# 1.) Input PSSM and FASTA Seq file of protein (assert protein seq is valid)
# 2.) User decides whether they will use pssm or seq or both
# 3.) user inputs model to predict on
# 4.) Load model
# 5.)

# import as pdb file???
# 
# download from pdb URL given pdb ID
