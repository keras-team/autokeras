class NERData():
    sents = []
    vocab = []
    uniq_tags = []
    sentnum = 1

    class Sentence():
        num = 0
        words = []
        tags = []

        def __init__(self, num):
            self.num = num
            self.words = []
            self.tags = []

        def printWords(self):
            print(self.words)

        def printTags(self):
            print(self.tags)

    def __init__(self):
        pass

    def loadData(self, filename):
        with open(filename) as f:
            lines = f.readlines()

        sent = self.Sentence(self.sentnum)
        for line in lines:
            if line != '\n':
                word, _, tag = line.strip().split()
                sent.words.append(word)
                sent.tags.append(tag)

                if word not in self.vocab:
                    self.vocab.append(word)
                
                if tag not in self.uniq_tags:
                    self.uniq_tags.append(tag)
            else:
                self.sents.append(sent)
                self.sentnum += 1
                sent = self.Sentence(self.sentnum)

    def printSentence(self, num):
        for sent in self.sents:
            if sent.num == num:
                print(sent.words)
                print(sent.tags)

    def printVocab(self):
        print(self.vocab)

    def printUniqTags(self):
        print(self.uniq_tags)
