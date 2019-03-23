# naive bayes class file
# to implement naive bayes algorithm to classify spam e-mails
# developer and owner of this program: Shrey S V (ssv170001)
# created by alice_v3.0.4 on 03/22/2019
# last modified on 03/23/2019

import os
import re
import io
import numpy as np

class NB:

    def __init__ (self, training_ham_path, training_spam_path, testing_ham_path, testing_spam_path, 
                    include_stop_words):
        self.training_ham_path = training_ham_path
        self.training_spam_path = training_spam_path
        self.testing_ham_path = testing_ham_path
        self.testing_spam_path = testing_spam_path
        self.vocabulary = []
        self.ham_word = {}
        self.spam_word = {}
        self.prior_ham = 0
        self.prior_spam = 0
        self.conditional_probability = {}
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

            for line in fopen.readlines():
                words = (re.sub ("[^a-zA-Z\s]", "", line)).lower().split()
                for word in words:
                    if not self.include_stop_words and word not in self.stop_words:
                        word_count [word] = word_count.get(word, 0) + 1
                    elif self.include_stop_words:
                        word_count [word] = word_count.get(word, 0) + 1
            fopen.close()
        
        return word_count
        
    def set_up_nb (self): 
        self.ham_word = self.generate_vocabulary (self.training_ham_path)
        self.spam_word = self.generate_vocabulary (self.training_spam_path)
        self.vocabulary = set (list (self.ham_word.keys()) + list (self.spam_word.keys()))
    
    def set_conditional_probability (self): 

        ham_count_sum = 0
        spam_count_sum = 0
        
        for key, value in self.ham_word.items():
            ham_count_sum += value
        ham_count_sum += len (list (self.ham_word.keys()))
        
        for key, value in self.spam_word.items():
            spam_count_sum += value
        spam_count_sum += len (list (self.spam_word.keys()))

        for word in self.vocabulary:
            ham_count = self.ham_word [word] + 1 if word in self.ham_word else 1
            spam_count = self.spam_word [word] + 1 if word in self.spam_word else 1
            
            self.conditional_probability [word] = [(float) (ham_count / ham_count_sum) , 
                                                    (float) (spam_count / spam_count_sum)]
        
    def train_nb (self): 
        ham_file_count = len (os.listdir (self.training_ham_path))
        spam_file_count = len (os.listdir (self.training_spam_path))

        self.prior_ham = ham_file_count / (ham_file_count + spam_file_count)
        self.prior_spam = spam_file_count / (ham_file_count + spam_file_count)
        self.set_conditional_probability()

    def log2 (self, x): 
        return np.log2(x) if x else 0
   
    def classify (self, flag):
        correct_predictions = 0
        testing_directory = self.testing_spam_path if flag else self.testing_ham_path

        for item in os.listdir(testing_directory): 

            fopen = io.open(testing_directory + "/" + item, 'r', encoding = 'iso-8859-1')
            data = fopen.read()
           
            words = set (((re.sub ("[^a-zA-Z\s]", "", data)).lower()).split())
            
            ham_score = self.log2 (self.prior_ham)
            for word in words:
                if word in self.vocabulary:
                    ham_score += self.log2 (self.conditional_probability [word][0])
                    
            spam_score = self.log2 (self.prior_spam)
            for word in words:
                if word in self.vocabulary:
                    spam_score += self.log2 (self.conditional_probability [word][1])

            if flag and spam_score > ham_score: correct_predictions += 1
            if not flag and ham_score > spam_score: correct_predictions += 1
            fopen.close()
        return correct_predictions
    
    def test_nb (self):       

        ham_predictions = self.classify (False)
        print("Ham Accuracy = ", (float) (ham_predictions / len (os.listdir (self.testing_ham_path))) * 100)

        spam_predictions = self.classify (True)
        print("spam Accuracy = ", (float) (spam_predictions / len (os.listdir (self.testing_spam_path))) * 100)
 
        print("Total Accuracy=", (float) ((ham_predictions + spam_predictions) /
                (len (os.listdir (self.testing_ham_path)) + 
                len (os.listdir (self.testing_spam_path)))) * 100)