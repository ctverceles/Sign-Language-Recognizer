import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    # general/specific algorithm snippets from forum discussion:
    # https://discussions.udacity.com/t/recognizer-implementation/234793/22
    # https://discussions.udacity.com/t/recognizer-implementation/234793/28

    hwords = test_set.get_all_Xlengths()
    for word_id in range(0, len(test_set.get_all_sequences())):
        # print('training on word_id '+str(word_id)+', current word is '+current_word)
        try:
            p_of_words = {}
            max_score = float("-inf")
            guess_word = None
            x, lengths = hwords[word_id]

            # for each model, get teh highest likelyhood (score) then record the respective word as guess_word
            # to add into guesses list
            for word, model in models.items():
                try:
                    score = model.score(x, lengths)
                    p_of_words[word] = score

                    if score > max_score:
                        guess_word = word
                        max_score = score
                except:
                    # fill in the probability dict if no probability is found
                    p_of_words[word] = float("-inf")
                    pass
        except:
            pass

        probabilities.append(p_of_words)
        guesses.append(guess_word)

    return probabilities, guesses
