import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def calculate_bic(self, state_num, l, n_features):
        p = state_num**2 + (2*state_num*n_features) - 1
        return (-2) * l + p * math.log(len(self.X))

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        score = float("inf")
        bestmodel = None

        # choose the model where likelyhood (bic) is lowest according to BIC equation above
        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(n)
                bic = self.calculate_bic(n, model.score(self.X, self.lengths), model.n_features)
                if bic < score:
                    score = bic
                    bestmodel = model
            except:
                continue

        return bestmodel


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    # separate sum calculation to make equation easier to read
    def calculate_sum(self, n):
        anti_logL = 0
        for word in self.words.keys():
            if word != self.this_word:
                x_other, lengths_other = self.hwords[word]
                anti_model = GaussianHMM(n, n_iter=1000).fit(x_other, lengths_other)
                anti_logL += anti_model.score(x_other, lengths_other)
        return anti_logL

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        score = float("-inf")
        bestmodel = None

        # choose the model where the likelyhood (dic) is highest according to DIC equation above
        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000, random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                dic = model.score(self.X, self.lengths) - self.calculate_sum(n)/(len(self.words)-1)
                if dic > score:
                    score = dic
                    bestmodel = model
            except:
                continue

        if bestmodel is None:
            bestmodel = self.base_model(self.n_constant)

        return bestmodel

class SelectorCV(ModelSelector):

    # put together with help from various forums topics on SelectorCV

    def select(self):
        #print('insideSelectorCV')
        #return 0
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score = float("-inf")
        bestmodel = None
        best_n = 0

        if len(self.sequences) > 1:
            split_method = KFold(n_splits=min(3, len(self.sequences)))

            # add 1 to account for range() counting
            for n in range(self.min_n_components, self.max_n_components + 1):
                # choose the model where the average likelyhood is highest in a given fold section
                try:
                    split_count = 1
                    total_score_splits = 0.0

                    for trainidx, testidx in split_method.split(self.sequences):
                        xtrain, lengthstrain = np.asarray(combine_sequences(trainidx, self.sequences))
                        xtest, lengthstest = np.asarray(combine_sequences(testidx, self.sequences))

                        model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000, random_state=self.random_state, verbose=False).fit(xtrain, lengthstrain)
                        logL = model.score(xtest, lengthstest)
                        total_score_splits += logL
                        split_count += 1

                    average_across_splits = total_score_splits/split_count
                    if average_across_splits > best_score:
                        best_score = average_across_splits
                        best_n = n
                except:
                    continue

                if best_n > 0:
                    bestmodel = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000, random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
        # handle situations where you need to use less than 3 folds
        else:
            # when less than 3 folds, choose the model where likelyhood is simply greatest
            for n in range(self.min_n_components, self.max_n_components + 1):
                try:
                    model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000, random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                    logL = model.score(self.X, self.lengths)
                    if logL > best_score:
                        best_score = logL
                        bestmodel = model
                except:
                    continue

        if bestmodel is None:
            bestmodel = GaussianHMM(n_components=self.n_constant, covariance_type="diag", n_iter=1000, random_state=self.random_state, verbose=False).fit(self.X, self.lengths)

        return bestmodel