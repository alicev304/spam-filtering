# logistic regression class file
# to implement logistic regression algorithm to classify spam e-mails
# developer and owner of this program: Shrey S V (ssv170001)
# created by alice_v3.0.4 on 03/22/2019
# last modified on 03/23/2019

import os
import re
import io
import numpy as np

#logistic regression
class LR:
    def __init__ (self, training_ham_path, training_spam_path, testing_ham_path, testing_spam_path, 
                    include_stop_words, learning_rate = 0.001, L = 10.0, iterations = 100):
        self.training_ham_path = training_ham_path
        self.training_spam_path = training_spam_path
        self.testing_ham_path = testing_ham_path
        self.testing_spam_path = testing_spam_path
        self.vocabulary = []
        self.ham_word = {}
        self.spam_word = {}
        self.file_word_count = {}
        self.tvalue = {}
        self.weights = {}
        self.error = {}
        self.learning_rate = learning_rate
        self.L = L
        self.iterations = iterations
        self.include_stop_words = include_stop_words
        self.stop_words = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", 
                            "any", "are", "arent", "as", "at", "be", "because", "been", "before", "being", 
                            "below", "between", "both", "but", "by", "cant", "cannot", "could", "couldnt", 
                            "did", "didnt", "do", "does", "doesnt", "doing", "dont", "down", "during", "each", 
                            "few", "for", "from", "further", "had", "hadnt", "has", "hasnt", "have", "havent", 
                            "having", "he", "hed", "hell", "hes", "her", "here", "heres", "hers", "herself", 
                            "him", "himself", "his", "how", "hows", "i", "id", "ill", "im", "ive", "if", "in", 
                            "into", "is", "isnt", "it", "its", "its", "itself", "lets", "me", "more", "most", 
                            "mustnt", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", 
                            "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", 
                            "shant", "she", "shed", "shell", "shes", "should", "shouldnt", "so", "some", "such", 
                            "than", "that", "thats", "the", "their", "theirs", "them", "themselves", "then", 
                            "there", "theres", "these", "they", "theyd", "theyll", "theyre", "theyve", "this", 
                            "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasnt", 
                            "we", "wed", "well", "were", "weve", "were", "werent", "what", "whats", "when", 
                            "whens", "where", "wheres", "which", "while", "who", "whos", "whom", "why", "whys", 
                            "with", "wont", "would", "wouldnt", "you", "youd", "youll", "youre", "youve", "your", 
                            "yours", "yourself", "yourselves"]
                       
    def generate_vocabulary (self, path):

        training_directory = os.listdir (path)
        word_count = {}
        for item in training_directory:
            fopen = io.open(path + "/" + item, 'r', encoding = 'iso-8859-1')
            self.file_word_count [item] = {}
            self.tvalue [item] = 1.0 if path == self.training_ham_path else 0.0
        
            for line in fopen.readlines():
                words = (re.sub ("[^a-zA-Z\s]", "", line)).lower().split()
                for word in words:
                    if not self.include_stop_words and word not in self.stop_words:
                        word_count [word] = word_count.get(word, 0) + 1
                        self.file_word_count [item][word] = self.file_word_count [item].get(word, 0) + 1
                    elif self.include_stop_words:
                        word_count [word] = word_count.get(word, 0) + 1
                        self.file_word_count [item][word] = self.file_word_count [item].get(word, 0) + 1
                        
            fopen.close()

        return word_count
        
    def set_up_lr (self):
        self.ham_word = self.generate_vocabulary (self.training_ham_path)
        self.spam_word = self.generate_vocabulary (self.training_spam_path)
        self.vocabulary = set (list (self.ham_word.keys()) + list (self.spam_word.keys()))
        for word in self.vocabulary:
            self.weights [word] = 0.0

    def sigmoid (self, x):
        return 1 / (1 + np.exp (-x))

    def update_errors (self):

        for item in os.listdir(self.training_ham_path):
            value = 1
            for word, count in self.file_word_count [item].items():
                value += count * self.weights [word]
            self.error [item] = self.sigmoid (value)
            
        for item in os.listdir(self.training_spam_path):
            value = 1
            for word, count in self.file_word_count [item].items():
                value += count * self.weights [word]
            self.error [item] = self.sigmoid (value)
    
    def update_weights (self):
        for word in self.vocabulary:
            value = 0
            for item in self.file_word_count.keys():
                if word in self.file_word_count [item]:
                    value += self.file_word_count [item][word] * (self.tvalue [item] - self.error [item])
            self.weights [word] += ((value * self.learning_rate) - 
                                        (self.learning_rate * self.L * self.weights [word]))
    
    def train_lr (self):
        for i in range (0, self.iterations):
            print ("Iteration: ", i)
            self.update_errors ()
            self.update_weights ()
    
    def classify (self, flag):
        correct_predictions = 0
        testing_directory = self.testing_spam_path if flag else self.testing_ham_path

        for item in os.listdir (testing_directory):
            value = 0
            word_count = {}
            fopen = io.open (testing_directory + "/" + item, 'r', encoding = 'iso-8859-1')
            
            for line in fopen.readlines():
                words = (re.sub("[^a-zA-Z\s]", "", line)).lower().split()
                for word in words:
                    if not self.include_stop_words and word not in self.stop_words:
                        word_count [word] = word_count.get(word, 0) + 1
                    elif self.include_stop_words:
                        word_count [word] = word_count.get(word, 0) + 1
            fopen.close()

            for word, count in word_count.items():
                if word in self.vocabulary:
                    value += self.weights [word] * count
            
            if flag and self.sigmoid (value) < 0.5: correct_predictions += 1
            if not flag and self.sigmoid (value) > 0.5: correct_predictions += 1

        return correct_predictions
    
    def test_lr (self):
        
        ham_predictions = self.classify (False)
        print("Ham Accuracy = ", (float) (ham_predictions / len (os.listdir (self.testing_ham_path))) * 100)

        spam_predictions = self.classify (True)
        print("spam Accuracy = ", (float) (spam_predictions / len (os.listdir (self.testing_spam_path))) * 100)
 
        print("Total Accuracy=", (float) ((ham_predictions + spam_predictions) /
                (len (os.listdir (self.testing_ham_path)) + 
                len (os.listdir (self.testing_spam_path)))) * 100)