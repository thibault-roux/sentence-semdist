from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def calculate_score(ref, hyp, model):
	refemb = model.encode(ref)
	hypemb = model.encode(hyp)

	print(cosine_similarity(refemb, hypemb)[0][0])


if __name__ == "__main__":
    model = SentenceTransformer('dangvantuan/sentence-camembert-large')

	while True:
		ref = input("Enter reference sentence: ")
		hyp = input("Enter hypothesis sentence: ")
		calculate_score(ref, hyp, model)