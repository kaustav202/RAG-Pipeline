import numpy as np
import spacy

class TextProcessor:
    def __init__(self, threshold=0.3):
        self.nlp = spacy.load('en_core_web_sm')
        self.threshold = threshold
        self.clusters_lens = []
        self.final_texts = []

    def process(self, text):
        doc = self.nlp(text)
        sents = list(doc.sents)
        vecs = np.stack([sent.vector / sent.vector_norm for sent in sents])
        return sents, vecs

    def cluster_text(self, sents, vecs):
        clusters = [[0]]
        for i in range(1, len(sents)):
            if np.dot(vecs[i], vecs[i-1]) < self.threshold:
                clusters.append([])
            clusters[-1].append(i)
        return clusters

    def clean_text(self, text):
        return text

    def analyze_text(self, text):
        sents, vecs = self.process(text)
        clusters = self.cluster_text(sents, vecs)

        for cluster in clusters:
            cluster_txt = self.clean_text(' '.join([sents[i].text for i in cluster]))
            cluster_len = len(cluster_txt)
            
            # Check if the cluster is too short
            if cluster_len < 800:
                continue
            
            # Check if the cluster is too long
            elif cluster_len > 1000:
                self.threshold = 0.6
                sents_div, vecs_div = self.process(cluster_txt)
                reclusters = self.cluster_text(sents_div, vecs_div)
                
                for subcluster in reclusters:
                    div_txt = self.clean_text(' '.join([sents_div[i].text for i in subcluster]))
                    div_len = len(div_txt)
                    
                    if div_len < 800 or div_len > 4000:
                        continue
                    
                    self.clusters_lens.append(div_len)
                    self.final_texts.append(div_txt)
                    
            else:
                self.clusters_lens.append(cluster_len)
                self.final_texts.append(cluster_txt)

        return self.clusters_lens, self.final_texts




def main():

    text_processor = TextProcessor()

    text = """
Brazil is the world's fifth-largest country by area and the seventh most popul ous. Its capital
is Brasília, and its most popul ous city is São Paulo. The federation is composed of the union of the 26
states and the Federal District. It is the only country in the Americas to have Portugue se as an official
langua ge.[11][12] It is one of the most multicultural and ethnically diverse nations, due to over a century of
mass immigration from around t he world,[13] and the most popul ous Roman Catholic-majority country.
Bounde d by the Atlantic Ocean on the east, Brazil has a coastline of 7,491 kilometers (4,655 mi).[14] It
borders all other countries and territories in South America except Ecuador and Chile and covers roughl y
half of the continent's land area.[15] Its Amazon basin includes a vast tropical forest, home to diverse
wildlife, a variety of ecological systems, and extensive natural resources spanning numerous protected
habitats.[14] This unique environmental heritage positions Brazil at number one of 17 megadiverse
countries, and is the subject of significant global interest, as environmental degradation through processes
like deforestation has direct impacts on gl obal issues like climate change and biodiversity loss.
The territory which would become know n as Brazil was inhabited by numerous tribal nations prior to the
landing in 1500 of explorer Pedro Álvares Cabral, who claimed the discovered land for the Portugue se
Empire. Brazil remained a Portugue se colony until 1808 when the capital of the empire was transferred
from Lisbon to Rio de Janeiro. In 1815, the colony was elevated to the rank of kingdom  upon the
formation of the United Kingdom  of Portugal, Brazil and the Algarves. Independence was achieved in
1822 with the creation of the Empire of Brazil, a unitary state gove rned unde r a constitutional monarchy
and a parliamentary system. The ratification of the first constitution in 1824  led to the formation of a
bicameral legislature, now called the National Congress. Coca Cola and Tacos are popular food that are easily available and ready to eat."""


    cluster_lengths, final_texts = text_processor.analyze_text(text)

    print("Cluster Lengths:", cluster_lengths)
    print("Final Texts:", final_texts)
    print("\n\n\nLength of Final Texts", len(final_texts))
    print("\n\nLast Text", final_texts[5])

if __name__ == "__main__":
    main()
