# driver file
# to classify spam e-mails using naive bayes or logistic regression algorithm
# developer and owner of this program: Shrey S V (ssv170001)
# created by alice_v3.0.4 on 03/22/2019
# last modified on 03/23/2019

import os
import nb
import lr

script_dir = os.path.dirname (__file__)
rel_training_ham_path = input ("Enter the path to training directory for ham files: ")
abs_training_ham_path = os.path.join (script_dir, rel_training_ham_path)
assert os.path.isdir(abs_training_ham_path), "Directory Does Not Exist"
rel_training_spam_path = input ("Enter the path to training directory for spam files: ")
abs_training_spam_path = os.path.join (script_dir, rel_training_spam_path)
assert os.path.isdir(abs_training_spam_path), "Directory Does Not Exist"
rel_testing_ham_path = input ("Enter the path to testing directory for ham files: ")
abs_testing_ham_path = os.path.join (script_dir, rel_testing_ham_path)
assert os.path.isdir(abs_testing_ham_path), "Directory Does Not Exist"
rel_testing_spam_path = input ("Enter the path to testing directory for ham files: ")
abs_testing_spam_path = os.path.join (script_dir, rel_testing_spam_path)
assert os.path.isdir(abs_testing_spam_path), "Directory Does Not Exist"
algorithm = input ("Select a learning algorithm: [NB or LR] ").lower()
assert algorithm == "nb" or algorithm == "lr", "Invlaid Input"
stop_words_flag = input ("Include stop words? [Y or N] ").lower()
assert stop_words_flag == "y" or stop_words_flag == "n", "Invlaid Input"
include_stop_words = True if stop_words_flag == "y" else False

if algorithm == "nb": 
    model = nb.NB(abs_training_ham_path,abs_training_spam_path,abs_testing_ham_path,abs_testing_spam_path, 
                    include_stop_words)
    model.set_up_nb()
    model.train_nb()
    model.test_nb()

else: 
    print ("Default values: Lambda = 10.0, Learning Rate = 0.01, Maximum Iterations = 100")
    default_flag = input ("Use default parameters? [Y or N] ").lower()
    assert default_flag == "y" or default_flag == "n", "Invlaid Input"
    if default_flag == "y":
        model = lr.LR(abs_training_ham_path,abs_training_spam_path,abs_testing_ham_path,abs_testing_spam_path, 
                    include_stop_words)
    else: 
        L = float (input ("Lambda          : "))
        learning_rate = float (input ("Learning Rate   : "))
        iterations = int (input ("Iterations     : "))
        model = lr.LR(abs_training_ham_path,abs_training_spam_path,abs_testing_ham_path,abs_testing_spam_path, 
                    include_stop_words, L, learning_rate, iterations)
    model.set_up_lr()
    model.train_lr()
    model.test_lr() 