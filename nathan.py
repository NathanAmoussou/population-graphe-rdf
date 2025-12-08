# Import du texte

with open("texte_a_analyser.txt", "r", encoding="utf-8") as f:
    text_source = f.read()

# Découpage : on sépare par les doubles sauts de ligne (\n\n) et on enlève les vides
blocs = [b.strip() for b in text_source.split('\n') if b.strip()]

print(f"Texte découpé avec succès en {len(blocs)} blocs (paragraphes).")
print(f"Exemple du bloc 1 : {blocs[0]}")

# REBEL

from transformers import pipeline
import pprint

# 1. Chargement du pipeline REBEL
# Le premier lancement téléchargera le modèle (~1.6 Go)
triplet_extractor = pipeline('text2text-generation', model='Babelscape/rebel-large', tokenizer='Babelscape/rebel-large')

# 2. Fonction minimale pour parser la sortie (String -> Liste de Dictionnaires)
def extract_triplets(text):
    triplets = []
    relation, subject, object_ = '', '', ''
    text = text.strip()
    current = 'x'
    # on enlève juste <s>, <pad>, </s> mais on garde <triplet>, <subj>, <obj>
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
    return triplets


# 3. Exécution sur les blocs
print("Démarrage de l'extraction REBEL...\n")
all_triplets = []

for i, bloc in enumerate(blocs):
    # 1) génération : on veut les ids, pas le texte nettoyé
    out = triplet_extractor(
        bloc,
        max_length=512,
        num_beams=3,
        return_tensors=True,
        return_text=False
    )[0]["generated_token_ids"]

    # 2) décodage manuel en gardant les tokens spéciaux
    raw_output = triplet_extractor.tokenizer.batch_decode(
        [out],
        skip_special_tokens=False
    )[0]

    # Debug si tu veux voir la chaîne brute
    print(f"RAW OUTPUT BLOC {i+1} : {repr(raw_output)}")

    # 3) extraction des triplets
    triplets = extract_triplets(raw_output)
    all_triplets.extend(triplets)

    print(f"Bloc {i+1} : {len(triplets)} triplets extraits.")

print(f"\nTerminé ! Total de triplets trouvés : {len(all_triplets)}")
pprint.pprint(all_triplets)

# Sortie :

# Démarrage de l'extraction REBEL...

# RAW OUTPUT BLOC 1 : '<s><triplet> Everest Hinton <subj> 6 December 1947 <obj> date of birth <subj> computer scientist <obj> occupation <subj> artificial neural network <obj> field of work <subj> University of Toronto <obj> employer <triplet> artificial neural network <subj> cognitive psychologist <obj> part of</s>'
# Bloc 1 : 5 triplets extraits.
# RAW OUTPUT BLOC 2 : '<s><triplet> AlexNet <subj> Alex Krizhevsky <obj> discoverer or inventor <triplet> Alex Krizhevsky <subj> AlexNet <obj> notable work</s>'
# Bloc 2 : 2 triplets extraits.
# RAW OUTPUT BLOC 3 : '<s><triplet> Christopher Longuet-Higgins <subj> University of Edinburgh <obj> employer</s>'
# Bloc 3 : 1 triplets extraits.
# RAW OUTPUT BLOC 4 : '<s><triplet> Google <subj> DNNresearch Inc. <obj> subsidiary <triplet> DNNresearch Inc. <subj> Google <obj> parent organization</s>'
# Bloc 4 : 2 triplets extraits.
# RAW OUTPUT BLOC 5 : '<s><triplet> neural network <subj> machine learning <obj> use</s>'
# Bloc 5 : 1 triplets extraits.
# RAW OUTPUT BLOC 6 : '<s><triplet> David E. Rumelhart <subj> Carnegie Mellon University <obj> educated at</s>'
# Bloc 6 : 1 triplets extraits.
# RAW OUTPUT BLOC 7 : '<s><triplet> Boltzmann machine <subj> David Ackley <obj> discoverer or inventor <subj> Terry Sejnowski <obj> discoverer or inventor</s>'
# Bloc 7 : 2 triplets extraits.
# RAW OUTPUT BLOC 8 : '<s><triplet> open-access <subj> research papers <obj> subclass of</s>'
# Bloc 8 : 1 triplets extraits.
# RAW OUTPUT BLOC 9 : '<s><triplet> Richard Zemel <subj> Brendan Frey <obj> student <triplet> Brendan Frey <subj> Richard Zemel <obj> student of</s>'
# Bloc 9 : 2 triplets extraits.

# Terminé ! Total de triplets trouvés : 17


# [
#   {'head': 'Hinton', 'type': 'works at', 'tail': 'Google'},
#   {'head': 'Hinton', 'type': 'born in', 'tail': '1947'}
# ]
