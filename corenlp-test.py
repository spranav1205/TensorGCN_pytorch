from stanfordcorenlp import StanfordCoreNLP

corenlp_path = r"C:\Users\spran\Desktop\TENSORGCN_PYTORCH\stanford-corenlp-4.5.10"

nlp = StanfordCoreNLP(corenlp_path, lang='en')

sentence = "Stanford University is located in California."
print("Tokens:", nlp.word_tokenize(sentence))
print("POS:", nlp.pos_tag(sentence))
print("NER:", nlp.ner(sentence))
print("Parse:", nlp.parse(sentence))
print("Dep Parse:", nlp.dependency_parse(sentence))

nlp.close()
